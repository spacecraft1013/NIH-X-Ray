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
from tqdm import tqdm

from multithreaded_preprocessing import PreprocessImages

MODEL_SAVE_NAME = "densenet201_pytorch"
EPOCHS = 250
IMAGE_SIZE = 256
BATCH_SIZE = 32
CHECKPOINT_DIR = f"data/checkpoints/{MODEL_SAVE_NAME}/"
NUM_GPUS = torch.cuda.device_count()
PIN_MEM = True


def init_process(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '15000'
    dist.init_process_group("nccl", world_size=world_size, rank=rank)


def train(rank, world_size, traindata, valdata, testdata,
          scaler, model, starttime, trainsampler, testsampler, valsampler):
    init_process(rank, world_size)

    model.to(rank)
    ddp_model = DDP(model, device_ids=rank)

    loss_fn = nn.MultiLabelMarginLoss().to(rank)
    optimizer = SGD(ddp_model.parameters(), lr=1e-6, momentum=0.9)
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer)

    best_model_wts = copy.deepcopy(ddp_model.state_dict())

    for epoch in range(EPOCHS):
        print(f"Epoch {epoch+1}/{EPOCHS}")
        print('='*61)

        trainsampler.set_epoch(epoch)
        valsampler.set_epoch(epoch)
        testsampler.set_epoch(epoch)

        running_loss = 0.0

        print('Training')
        progressbar = tqdm(traindata, unit='steps', dynamic_ncols=True)
        for index, (inputs, labels) in enumerate(progressbar):

            inputs, labels = inputs.to(rank), labels.to(rank)

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
            progressbar.set_description(f'Loss: {running_loss/(index+1):.5f}')
            progressbar.refresh()

            scheduler.step(loss)

        running_loss = 0.0

        ddp_model.eval()
        print('Validation')
        progressbar = tqdm(traindata, unit='steps', dynamic_ncols=True)
        for index, (inputs, labels) in enumerate(progressbar):

            inputs, labels = inputs.to(rank), labels.to(rank)

            with torch.no_grad():
                with autocast():
                    outputs = ddp_model(inputs)
                    loss = loss_fn(outputs, labels.long())

            running_loss += loss.item() * inputs.size(0)
            progressbar.set_description(f'Loss: {running_loss/(index+1):.5f}')
            progressbar.refresh()

        val_loss = running_loss / len(valdata)

        if epoch == 0:
            best_loss = val_loss
            best_model_wts = copy.deepcopy(ddp_model.state_dict())
        elif val_loss < best_loss:
            best_loss = val_loss
            best_model_wts = copy.deepcopy(ddp_model.state_dict())

        checkpoint_path = os.path.join(CHECKPOINT_DIR,
                                       f"checkpoint-{epoch:03d}.pth")
        if rank == 0:
            torch.save(ddp_model.state_dict(), checkpoint_path)
            print(f"Checkpoint saved to checkpoint-{epoch:03d}.pth\n")

        dist.barrier()
        map_location = {'cuda:0': f'cuda:{rank}'}
        ddp_model.load_state_dict(torch.load(checkpoint_path,
                                             map_location=map_location))

    time_elapsed = time.time() - starttime
    print(f"Training complete in {time_elapsed // 60}m {time_elapsed % 60}s")

    ddp_model.load_state_dict(best_model_wts)

    ddp_model.eval()
    running_loss = 0.0
    print('Testing')
    progressbar = tqdm(testdata, unit='steps', dynamic_ncols=True)
    for index, (inputs, labels) in enumerate(progressbar):

        inputs, labels = inputs.to(rank), labels.to(rank)

        with torch.no_grad():
            with autocast():
                outputs = ddp_model(inputs)
                loss = loss_fn(outputs, labels.long())

        running_loss += loss.item() * inputs.size(0)
        progressbar.set_description(f'Test loss: {running_loss/(index+1):.5f}')
        progressbar.refresh()

    if rank == 0:
        print("Saving model weights")
        savepath = f"data/models/{MODEL_SAVE_NAME}_weights.pth"
        torch.save(ddp_model.state_dict(), savepath)
        print("Model saved!\n")

        print("Saving ONNX file")
        dummy_input = torch.randn(1, 1, IMAGE_SIZE, IMAGE_SIZE, device='cuda')
        torch.onnx.export(ddp_model, dummy_input, savepath, verbose=True)
        onnx.checker.check_model(f'data/models/{MODEL_SAVE_NAME}.onnx')
        print("ONNX model has been successfully saved!")


if __name__ == '__main__':
    print("Importing Arrays")
    if not os.path.exists(f"data/arrays/X_train_{IMAGE_SIZE}.npy"):
        print("Arrays not found, generating...")
        preprocessor = PreprocessImages("F:/Datasets/NIH X-Rays/data",
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
    model.classifier = nn.Linear(in_features=1920, out_features=15, bias=True)

    X_train = torch.Tensor(X_train)
    y_train = torch.Tensor(y_train)
    X_test = torch.Tensor(X_test)
    y_test = torch.Tensor(y_test)

    dataset = TensorDataset(X_train, y_train)
    train_set, val_set = random_split(dataset, [70000, 16524])
    test_set = TensorDataset(X_test, y_test)

    trainsampler = DistributedSampler(train_set, num_replicas=NUM_GPUS)
    valsampler = DistributedSampler(val_set, num_replicas=NUM_GPUS)
    testsampler = DistributedSampler(test_set, num_replicas=NUM_GPUS)

    traindata = DataLoader(train_set, shuffle=True, pin_memory=PIN_MEM,
                           batch_size=BATCH_SIZE, sampler=trainsampler)
    valdata = DataLoader(val_set, shuffle=True, pin_memory=PIN_MEM,
                         batch_size=BATCH_SIZE, sampler=valsampler)
    testdata = DataLoader(test_set, shuffle=True, pin_memory=PIN_MEM,
                          batch_size=BATCH_SIZE, sampler=testsampler)

    scaler = GradScaler()

    starttime = time.time()

    args = (NUM_GPUS, traindata, valdata, testdata, scaler, model, starttime,
            trainsampler, valsampler, testsampler)
    mp.spawn(train, args=args, nprocs=NUM_GPUS)
