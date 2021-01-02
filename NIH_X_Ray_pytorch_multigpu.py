import argparse
import copy
import os
import time

import numpy as np
import onnx
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import SGD, lr_scheduler
from torch.utils.data import DataLoader, TensorDataset, random_split
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter

from multithreaded_preprocessing import PreprocessImages

MODEL_SAVE_NAME = "densenet201_pytorch"
IMAGE_SIZE = 256
BATCH_SIZE = 32
CHECKPOINT_DIR = f"data/checkpoints/{MODEL_SAVE_NAME}-{int(time.time())}/"
NUM_GPUS = torch.cuda.device_count()
PIN_MEM = True

def train(gpu_num, scaler, model, starttime, train_set, val_set, test_set, args):

    rank = args.nr * args.gpus + gpu_num
    dist.init_process_group(
        backend='nccl',
        world_size=args.world_size,
        rank=rank
    )

    model.to(gpu_num)
    ddp_model = DDP(model, device_ids=[gpu_num])

    if args.checkpoint:
        ddp_model.load_state_dict(torch.load(args.checkpoint))

    trainsampler = DistributedSampler(train_set, num_replicas=NUM_GPUS)
    valsampler = DistributedSampler(val_set, num_replicas=NUM_GPUS)
    testsampler = DistributedSampler(test_set, num_replicas=NUM_GPUS)

    traindata = DataLoader(train_set, pin_memory=PIN_MEM,
                           batch_size=BATCH_SIZE, sampler=trainsampler)
    valdata = DataLoader(val_set, pin_memory=PIN_MEM,
                         batch_size=BATCH_SIZE, sampler=valsampler)
    testdata = DataLoader(test_set, pin_memory=PIN_MEM,
                          batch_size=BATCH_SIZE, sampler=testsampler)

    if rank == 0:
        writer = SummaryWriter("data/logs", comment=MODEL_SAVE_NAME)
        dummy_input = torch.randn(1, 1, IMAGE_SIZE, IMAGE_SIZE, device='cuda:0')
        writer.add_graph(model, dummy_input)
        writer.flush()

    loss_fn = nn.MultiLabelMarginLoss().to(gpu_num)
    optimizer = SGD(ddp_model.parameters(), lr=1e-6, momentum=0.9)
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer)
    mse_fn = nn.MSELoss().to(gpu_num)

    best_model_wts = copy.deepcopy(ddp_model.state_dict())

    for epoch in range(args.epochs):
        if gpu_num == 0:
            print(f"Epoch {epoch+1}/{args.epochs}")
            print('='*61)

        trainsampler.set_epoch(epoch)
        valsampler.set_epoch(epoch)
        testsampler.set_epoch(epoch)

        running_loss = 0.0
        running_mse = 0.0

        if gpu_num == 0:
            print('Training')
        for index, (inputs, labels) in enumerate(traindata):

            steptime = time.time()
            inputs, labels = inputs.to(gpu_num), labels.to(gpu_num)

            ddp_model.train()
            optimizer.zero_grad()

            with torch.set_grad_enabled(True):
                with autocast():
                    outputs = ddp_model(inputs)
                    loss = loss_fn(outputs, labels.long())
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

            running_loss += loss.item() * inputs.size(0)

            mse = mse_fn(outputs, labels.long())
            running_mse += mse.item() * inputs.size(0)
            if gpu_num == 0:
                print(f'{index+1}/{len(traindata)} Loss: {running_loss/(index+1):.5f}, \
MSE: {running_mse/(index+1):.5f}, {(time.time()-steptime)*1000:.2f}ms/step', end='\r')

            scheduler.step(loss)
        epoch_loss = running_loss / len(traindata)
        epoch_mse = running_mse / len(traindata)
        print()

        running_loss = 0.0
        running_mse = 0.0

        ddp_model.eval()
        if gpu_num == 0:
            print('Validation')
        for index, (inputs, labels) in enumerate(valdata):

            steptime = time.time()
            inputs, labels = inputs.to(rank), labels.to(rank)

            with torch.no_grad():
                with autocast():
                    outputs = ddp_model(inputs)
                    loss = loss_fn(outputs, labels.long())

            running_loss += loss.item() * inputs.size(0)

            mse = mse_fn(outputs, labels.long())
            running_mse += mse.item() * inputs.size(0)
            if gpu_num == 0:
                print(f'{index+1}/{len(valdata)} Loss: {running_loss/(index+1):.5f}, \
MSE: {running_mse/(index+1):.5f}, {(time.time()-steptime)*1000:.2f}ms/step', end='\r')
        print()

        val_loss = running_loss / len(valdata)
        val_mse = running_mse / len(valdata)

        if epoch == 0:
            best_loss = val_loss
            best_model_wts = copy.deepcopy(ddp_model.state_dict())
        elif val_loss < best_loss:
            best_loss = val_loss
            best_model_wts = copy.deepcopy(ddp_model.state_dict())

        if rank == 0:
            writer.add_scalars('Loss', {'Training': epoch_loss, 'Validation': val_loss}, epoch+1)
            writer.add_scalars('MSE', {'Training': epoch_mse, 'Validation': val_mse}, epoch+1)
            writer.flush()

        checkpoint_path = os.path.join(CHECKPOINT_DIR,
                                       f"checkpoint-{epoch:03d}.pth")
        if rank == 0:
            torch.save(ddp_model.state_dict(), checkpoint_path)
            print(f"Checkpoint saved to checkpoint-{epoch:03d}.pth\n")

        dist.barrier()
        map_location = {'cuda:0': f'cuda:{gpu_num}'}
        ddp_model.load_state_dict(torch.load(checkpoint_path,
                                             map_location=map_location))

    if rank == 0:
        writer.close()
    time_elapsed = time.time() - starttime
    print(f"Training complete in {time_elapsed // 3600}h \
{time_elapsed // 60}m {round(time_elapsed % 60)}s")

    ddp_model.load_state_dict(best_model_wts)

    ddp_model.eval()
    running_loss = 0.0

    if gpu_num == 0:
        print('Testing')
    for index, (inputs, labels) in enumerate(testdata):

        steptime = time.time()
        inputs, labels = inputs.to(rank), labels.to(rank)

        with torch.no_grad():
            with autocast():
                outputs = ddp_model(inputs)
                loss = loss_fn(outputs, labels.long())

        running_loss += loss.item() * inputs.size(0)
        if gpu_num == 0:
            print(f'{index+1}/{len(testdata)} Loss: {running_loss/(index+1):.5f}, {(time.time()-steptime)*1000:.2f}ms/step', end='\r')
    print()

    if rank == 0:
        print("Saving model weights")
        savepath = f"data/models/{MODEL_SAVE_NAME}-{int(time.time())}.pth"
        savepath_weights = f"data/models/{MODEL_SAVE_NAME}-{int(time.time())}_weights.pth"
        torch.save(ddp_model.state_dict(), savepath_weights)
        torch.save(ddp_model, savepath)
        print("Model saved!\n")

        print("Saving ONNX file")
        savepath_onnx = f"data/models/{MODEL_SAVE_NAME}-{int(time.time())}.onnx"
        dummy_input = torch.randn(1, 1, IMAGE_SIZE, IMAGE_SIZE, device='cuda:0')
        torch.onnx.export(ddp_model, dummy_input, savepath_onnx)
        onnx.checker.check_model(savepath_onnx)
        print("ONNX model has been successfully saved!")

    dist.destroy_process_group()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--nodes', default=1, type=int, metavar='N',
                        help='Number of nodes')
    parser.add_argument('-g', '--gpus', default=1, type=int,
                        help='Number of gpus per node')
    parser.add_argument('-nr', '--nr', default=0, type=int,
                        help='Node rank')
    parser.add_argument('-m', '--master', default='localhost', type=str,
                        help='IP address of master node')
    parser.add_argument('-p', '--port', default='15000', type=str,
                        help='Port to communicate over')
    parser.add_argument('-c', '--checkpoint', default=None, type=str,
                        help='Checkpoint file to load from')
    parser.add_argument('-e', '--epochs', default=250, type=int,
                        help='Number of epochs to use')
    args = parser.parse_args()
    args.world_size = args.gpus * args.nodes

    os.environ['MASTER_ADDR'] = args.master
    os.environ['MASTER_PORT'] = args.port

    print("Importing Arrays")
    if not os.path.exists(f"data/arrays/X_train_{IMAGE_SIZE}.npy"):
        print("Arrays not found, generating...")
        preprocessor = PreprocessImages("/data/ambouk3/NIH-X-Ray-Dataset",
                                        IMAGE_SIZE)
        (X_train, y_train), (X_test, y_test) = preprocessor()

    else:
        X_train = np.load(open(f"data/arrays/X_train_{IMAGE_SIZE}.npy", "rb"))
        y_train = np.load(open(f"data/arrays/y_train_{IMAGE_SIZE}.npy", "rb"))
        X_test = np.load(open(f"data/arrays/X_test_{IMAGE_SIZE}.npy", "rb"))
        y_test = np.load(open(f"data/arrays/y_test_{IMAGE_SIZE}.npy", "rb"))

    if not os.path.exists(CHECKPOINT_DIR):
        os.mkdir(CHECKPOINT_DIR)

    # Convert channels-last to channels-first format
    X_train = np.transpose(X_train, (0, 3, 1, 2))
    X_test = np.transpose(X_test, (0, 3, 1, 2))

    model = torch.hub.load('pytorch/vision:v0.6.0', 'densenet201',
                           pretrained=False)
    model.features[0] = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2),
                                  padding=(3, 3), bias=False)
    model.classifier = nn.Sequential(
        nn.Linear(in_features=1920, out_features=15, bias=True),
        nn.Sigmoid()
    )

    X_train = torch.Tensor(X_train)
    y_train = torch.Tensor(y_train)
    X_test = torch.Tensor(X_test)
    y_test = torch.Tensor(y_test)

    dataset = TensorDataset(X_train, y_train)
    train_set, val_set = random_split(dataset, [70000, 16524])
    test_set = TensorDataset(X_test, y_test)

    scaler = GradScaler()

    starttime = time.time()

    args = (scaler, model, starttime, train_set, val_set, test_set, args)
    mp.spawn(train, args=args, nprocs=NUM_GPUS, join=True)
