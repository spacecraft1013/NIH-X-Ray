import argparse
import copy
import os
import time

import numpy as np
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


def train(gpu_num, scaler, model, starttime,
          train_set, val_set, test_set, args):

    rank = args.node_rank * args.num_gpus + gpu_num
    dist.init_process_group(
        backend='nccl',
        world_size=args.world_size,
        rank=rank
    )

    torch.manual_seed(args.seed)

    if args.devices:
        model.to(args.devices[gpu_num])
        ddp_model = DDP(model, device_ids=[args.devices[gpu_num]])
    else:
        model.to(gpu_num)
        ddp_model = DDP(model, device_ids=[gpu_num])

    if args.checkpoint:
        ddp_model.load_state_dict(torch.load(args.checkpoint))

    trainsampler = DistributedSampler(train_set, num_replicas=args.num_gpus)
    valsampler = DistributedSampler(val_set, num_replicas=args.num_gpus)
    testsampler = DistributedSampler(test_set, num_replicas=args.num_gpus)

    traindata = DataLoader(train_set, pin_memory=(not args.no_pin_mem), drop_last=True,
                           batch_size=args.batch_size, sampler=trainsampler)
    valdata = DataLoader(val_set, pin_memory=(not args.no_pin_mem), drop_last=True,
                         batch_size=args.batch_size, sampler=valsampler)
    testdata = DataLoader(test_set, pin_memory=(not args.no_pin_mem), drop_last=True,
                          batch_size=args.batch_size, sampler=testsampler)

    if rank == 0:
        writer = SummaryWriter(args.log_dir)
        size_tuple = (1, 1, args.img_size, args.img_size)
        dummy_input = torch.randn(*size_tuple, device='cuda:0')
        writer.add_graph(model, dummy_input)
        writer.flush()

    loss_fn = nn.MultiLabelSoftMarginLoss().to(gpu_num)
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
                    mse = mse_fn(outputs, labels)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

            running_loss += loss.item()
            running_mse += mse.item()
            if gpu_num == 0:
                print(f'{index+1}/{len(traindata)} Loss: {running_loss/(index+1):.5f}, \
MSE: {running_mse/(index+1):.5f}, \
{(time.time()-steptime)*1000:.2f}ms/step', end='\r')

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
                    mse = mse_fn(outputs, labels)

            running_loss += loss.item()
            running_mse += mse.item()
            if gpu_num == 0:
                print(f'{index+1}/{len(valdata)} Loss: {running_loss/(index+1):.5f}, \
MSE: {running_mse/(index+1):.5f}, \
{(time.time()-steptime)*1000:.2f}ms/step', end='\r')
        print()

        val_loss = running_loss / len(valdata)
        val_mse = running_mse / len(valdata)
        scheduler.step(val_mse)

        if 'best_loss' not in locals():
            best_loss = val_loss

        if val_loss < best_loss:
            best_loss = val_loss
            best_model_wts = copy.deepcopy(model.state_dict())
            path = os.path.join(args.checkpoint_dir, 'best_weights.pth')
            torch.save(best_model_wts, path)

        if rank == 0:
            loss_data = {'Training': epoch_loss, 'Validation': val_loss}
            mse_data = {'Training': epoch_mse, 'Validation': val_mse}
            writer.add_scalars('Loss', loss_data, epoch+1)
            writer.add_scalars('MSE', mse_data, epoch+1)
            writer.flush()

        checkpoint_path = os.path.join(args.checkpoint_dir,
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
    time_hours = int(time_elapsed // 3600)
    time_minutes = int((time_elapsed-time_hours) // 60)
    time_seconds = round(time_elapsed % 60)
    print(f"Training complete in {time_hours}h \
    {time_minutes}m {time_seconds}s")

    ddp_model.load_state_dict(best_model_wts)

    ddp_model.eval()
    running_loss = 0.0
    running_mse = 0.0

    if gpu_num == 0:
        print('Testing')
    for index, (inputs, labels) in enumerate(testdata):

        steptime = time.time()
        inputs, labels = inputs.to(rank), labels.to(rank)

        with torch.no_grad():
            with autocast():
                outputs = ddp_model(inputs)
                loss = loss_fn(outputs, labels)
                mse = mse_fn(outputs, labels)

        running_loss += loss.item()
        running_mse += mse.item()
        if gpu_num == 0:
            print(f'{index+1}/{len(testdata)} Loss: {running_loss/(index+1):.5f}, \
MSE: {running_mse/(index+1):.5f}, \
{(time.time()-steptime)*1000:.2f}ms/step', end='\r')
    print()

    if rank == 0:
        print("Saving model weights")
        savepath = f"data/models/{args.name}-{int(starttime)}.pth"
        savepath_weights = f"data/models/{args.name}-{int(starttime)}_weights.pth"
        torch.save(ddp_model.state_dict(), savepath_weights)
        torch.save(ddp_model, savepath)
        print("Model saved!\n")

    dist.destroy_process_group()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--nodes', default=1, type=int, metavar='N',
                        help='Number of nodes')
    parser.add_argument('-g', '--num-gpus', default=1, type=int,
                        help='Number of gpus per node')
    parser.add_argument('--resume-latest', default=False, action='store_true',
                        help='Resume from latest checkpoint')
    parser.add_argument('-nr', '--node-rank', default=0, type=int,
                        help='Node rank')
    parser.add_argument('-m', '--master', default='localhost', type=str,
                        help='IP address of master node')
    parser.add_argument('-p', '--port', default='15000', type=str,
                        help='Port to communicate over')
    parser.add_argument('-c', '--checkpoint', default=None, type=str,
                        help='Checkpoint file to load from')
    parser.add_argument('-e', '--epochs', default=250, type=int,
                        help='Number of epochs to use')
    parser.add_argument('--no-pin-mem', default=False, action='store_true',
                        help="Don't use pinned memory")
    parser.add_argument('--name', default='model', type=str,
                        help='Name to save model (no file extension)')
    parser.add_argument('--checkpoint-dir', default=None, type=str,
                        help='Checkpoint directory')
    parser.add_argument('--log-dir', default=None, type=str,
                        help='Logging directory')
    parser.add_argument('--img-size', default=256, type=int,
                        help='Single sided image resolution')
    parser.add_argument('--batch-size', default=32, type=int,
                        help='Batch size to use for training')
    parser.add_argument('-s', '--seed', default=0, type=int,
                        help='Seed to use for random values')
    parser.add_argument('--devices', default=None, nargs='+',
                        help='GPU IDs to train on')
    args = parser.parse_args()

    args.world_size = args.num_gpus * args.nodes
    args.starting_epoch = None

    starttime = time.time()

    mp.set_sharing_strategy('file_system')

    if args.devices:
        args.num_gpus = len(args.devices)

    os.environ['MASTER_ADDR'] = args.master
    os.environ['MASTER_PORT'] = args.port

    print("Importing Arrays")
    if not os.path.exists(f"data/arrays/X_train_{args.img_size}.npy"):
        print("Arrays not found, generating...")
        preprocessor = PreprocessImages("/data/ambouk3/NIH-X-Ray-Dataset",
                                        args.img_size)
        (X_train, y_train), (X_test, y_test) = preprocessor()

    else:
        arrays = np.load(f"data/arrays/arrays_{args.img_size}.npz")
        X_train = arrays['X_train']
        y_train = arrays['y_train']
        X_test = arrays['X_test']
        y_test = arrays['y_test']

    if not args.checkpoint_dir:
        args.checkpoint_dir = f"data/checkpoints/{args.name}-{int(starttime)}/"

    if args.resume_latest:
        checkpoint_dirs = os.listdir("data/checkpoints")
        checkpoint_dirs = [i for i in checkpoint_dirs if i[-10:].isdigit()]
        timestamps = [int(i[-10:]) for i in checkpoint_dirs]
        max_index = timestamps.index(max(timestamps))
        args.checkpoint_dir = f"data/checkpoints/{checkpoint_dirs[max_index]}/"
        checkpoints = os.listdir(args.checkpoint_dir)
        checkpoint_nums = []
        for checkpoint in checkpoints:
            temp = []
            for i in checkpoint:
                if i.isdigit():
                    temp.append(i)
            if temp:
                checkpoint_nums.append(int(''.join(temp)))
        max_index = checkpoint_nums.index(max(checkpoint_nums))
        args.starting_epoch = max(checkpoint_nums)+1
        args.checkpoint = os.path.join(args.checkpoint_dir, checkpoints[max_index])

        log_dirs = os.listdir("data/logs")
        log_dirs = [i for i in log_dirs if i[-10:].isdigit()]
        timestamps = [int(i[-10:]) for i in log_dirs]
        max_index = timestamps.index(max(timestamps))
        args.log_dir = f"data/logs/{log_dirs[max_index]}/"

    if not os.path.exists(args.checkpoint_dir):
        os.mkdir(args.checkpoint_dir)

    # Convert channels-last to channels-first format
    X_train = np.transpose(X_train, (0, 3, 1, 2))
    X_test = np.transpose(X_test, (0, 3, 1, 2))

    model = torch.hub.load('pytorch/vision:v0.6.0', 'densenet201',
                           pretrained=False)
    model.features[0] = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2),
                                  padding=(3, 3), bias=False)
    model.classifier = nn.Sequential(
        nn.Linear(in_features=1920, out_features=14, bias=True),
        nn.Sigmoid()
    )

    X_train = torch.Tensor(X_train)
    y_train = torch.Tensor(y_train)
    X_test = torch.Tensor(X_test)
    y_test = torch.Tensor(y_test)

    dataset = TensorDataset(X_train, y_train)
    train_num = int(len(dataset)*0.7)
    train_set, val_set = random_split(dataset, [train_num, len(dataset)-train_num])
    test_set = TensorDataset(X_test, y_test)

    scaler = GradScaler()

    func_args = (scaler, model, starttime, train_set, val_set, test_set, args)
    mp.spawn(train, args=func_args, nprocs=args.num_gpus, join=True)
