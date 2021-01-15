import argparse
import os
import time
from functools import partial

import numpy as np
import ray
import torch
import torch.nn as nn
from ray.tune.integration.torch import (DistributedTrainableCreator,
                                        distributed_checkpoint_dir)
from ray.tune.schedulers import ASHAScheduler
from torch import optim
from torch.cuda.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, TensorDataset, random_split
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('-e', '--epochs', default=250, type=int,
                    help='Number of epochs to use')
parser.add_argument('--no-pin-mem', default=False, action='store_true',
                    help="Don't use pinned memory")
parser.add_argument('--name', default='model_raytune', type=str,
                    help='Name to save model (no file extension)')
parser.add_argument('--img-size', default=256, type=int,
                    help='Single sided image resolution')
parser.add_argument('-s', '--seed', default=0, type=int,
                    help='Seed to use for random values')
parser.add_argument('-n', '--num-workers', default=1, type=int,
                    help='Number of workers per training sample')
parser.add_argument('--num-trials', default=100, type=int,
                    help='Number of trials to run')
args = parser.parse_args()

starttime = time.time()

args.pin_mem = not args.no_pin_mem

torch.manual_seed(args.seed)


def train(config, args, checkpoint_dir=None):

    X_train = np.load(open(f"data/arrays/X_train_{args.img_size}.npy", "rb"))
    y_train = np.load(open(f"data/arrays/y_train_{args.img_size}.npy", "rb"))

    X_train = np.transpose(X_train, (0, 3, 1, 2))

    X_train = torch.Tensor(X_train)
    y_train = torch.Tensor(y_train)

    device = torch.device("cuda")

    dataset = TensorDataset(X_train, y_train)
    train_set, val_set = random_split(dataset, [70000, 16524])

    model = torch.hub.load('pytorch/vision:v0.6.0', 'densenet201',
                        pretrained=False)
    model.features[0] = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2),
                                padding=(3, 3), bias=False)
    model.classifier = nn.Sequential(
        nn.Linear(in_features=1920, out_features=15, bias=True),
        nn.Sigmoid()
    )

    lr = config['lr']

    model.to(device)
    model = DDP(model, device_ids=[device])

    loss_fn = nn.MultiLabelSoftMarginLoss().to(device)
    if config['optimizer'] == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    elif config['optimizer'] == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=lr)
    elif config['optimizer'] == 'adamax':
        optimizer = optim.Adamax(model.parameters(), lr=lr)
    mse_fn = nn.MSELoss()

    if checkpoint_dir:
        checkpoint = os.path.join(checkpoint_dir, "checkpoint")
        model_state, optimizer_state = torch.load(checkpoint)
        model.load_state_dict(model_state)
        optimizer.load_state_dict(optimizer_state)

    trainsampler = DistributedSampler(train_set, num_replicas=args.num_workers)
    valsampler = DistributedSampler(val_set, num_replicas=args.num_workers)

    traindata = DataLoader(train_set, pin_memory=args.pin_mem,
                           batch_size=config['batch_size'], sampler=trainsampler)
    valdata = DataLoader(val_set, pin_memory=args.pin_mem,
                         batch_size=config['batch_size'], sampler=valsampler)

    scaler = GradScaler()
    epoch = 0
    while True:
        print(f"Epoch {epoch+1}")
        running_loss = 0.0
        running_mse = 0.0
        print("Training")
        for inputs, labels in traindata:

            inputs, labels = inputs.to(device), labels.to(device)

            model.train()
            optimizer.zero_grad()

            with torch.set_grad_enabled(True):
                with autocast():
                    outputs = model(inputs)
                    loss = loss_fn(outputs, labels)
                    mse = mse_fn(outputs, labels)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

            running_loss += loss.item()
            running_mse += mse.item()

        running_loss = 0.0
        running_mse = 0.0

        print("Validating")
        model.eval()
        for inputs, labels in valdata:

            inputs, labels = inputs.to(device), labels.to(device)

            with torch.no_grad():
                with autocast():
                    outputs = model(inputs)
                    loss = loss_fn(outputs, labels)
                    mse = mse_fn(outputs, labels)

            running_loss += loss.item()
            running_mse += mse.item()

        val_loss = running_loss / len(valdata)
        val_mse = running_mse / len(valdata)

        epoch += 1

        with distributed_checkpoint_dir(epoch) as checkpoint_dir:
            path = os.path.join(checkpoint_dir, "checkpoint")
            torch.save((model.state_dict(), optimizer.state_dict()), path)

        ray.tune.report(loss=val_loss, mse=val_mse, epoch=epoch)


config = {
    'lr': ray.tune.loguniform(1e-8, 1e-1),
    'batch_size': ray.tune.choice([2, 4, 8, 16, 32, 64]),
    'optimizer': ray.tune.choice(['sgd', 'adam', 'adamax'])
}

scheduler = ASHAScheduler(
    metric="mse",
    mode='min',
    max_t=args.epochs,
    grace_period=1,
    reduction_factor=2)
distributed_cls = DistributedTrainableCreator(
    partial(train, args=args),
    num_workers=args.num_workers,
    num_cpus_per_worker=8,
    num_gpus_per_worker=1,
    backend='nccl')
analysis = ray.tune.run(
    distributed_cls,
    config=config,
    num_samples=args.num_trials,
    scheduler=scheduler,
    local_dir=f'data/logs/raytune-{int(starttime)}',
    verbose=3
)

best_trial = analysis.get_best_trial()
print(f"Best config: {best_trial.config}")
print(f"Best MSE: {best_trial.last_result['mse']}")
print(f"Best loss: {best_trial.last_result['loss']}")
print(f"Best trial log directory: {analysis.get_best_logdir()}")

running_mse = 0.0
mse_fn = nn.MSELoss()

X_test = np.load(open(f"data/arrays/X_test_{args.img_size}.npy", "rb"))
y_test = np.load(open(f"data/arrays/y_test_{args.img_size}.npy", "rb"))

X_test = torch.Tensor(X_test)
y_test = torch.Tensor(y_test)

model = torch.hub.load('pytorch/vision:v0.6.0', 'densenet201',
                       pretrained=False)
model.features[0] = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2),
                              padding=(3, 3), bias=False)
model.classifier = nn.Sequential(
    nn.Linear(in_features=1920, out_features=15, bias=True),
    nn.Sigmoid()
)

best_weights = torch.load(os.path.join(best_trial.checkpoint.value, "checkpoint"))
model.load_state_dict(best_weights)

model.eval()

test_set = TensorDataset(X_test, y_test)
testdata = DataLoader(test_set, shuffle=True,
                      pin_memory=args.pin_mem, batch_size=args.batch_size)

print('Testing')
progressbar = tqdm(testdata, dynamic_ncols=True)
for index, (inputs, labels) in enumerate(progressbar):

    inputs, labels = inputs.to(args.device), labels.to(args.device)

    with torch.no_grad():
        with autocast():
            outputs = model(inputs)
            mse = mse_fn(outputs, labels)

    running_mse += mse.item() * inputs.size(0)
    progressbar.set_description(f'Test MSE: {running_mse/(index+1):.5f}')
    progressbar.refresh()

print("Saving trial data")
analysis.results_df.to_csv(f'data/logs/raytune-{int(starttime)}/results.csv')

print("Saving model weights")
savepath = f"data/models/{args.name}-{int(starttime)}.pth"
savepath_weights = f"data/models/{args.name}-{int(starttime)}_weights.pth"
torch.save(model.state_dict(), savepath_weights)
torch.save(model, savepath)
print("Model saved!\n")
