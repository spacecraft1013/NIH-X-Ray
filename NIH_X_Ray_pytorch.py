import argparse
import copy
import os
import time

import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.optim import SGD, lr_scheduler
from torch.utils.data import DataLoader, TensorDataset, random_split
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from multithreaded_preprocessing import PreprocessImages

parser = argparse.ArgumentParser()
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
parser.add_argument('--img-size', default=256, type=int,
                    help='Single sided image resolution')
parser.add_argument('--batch-size', default=32, type=int,
                    help='Batch size to use for training')
parser.add_argument('-s', '--seed', default=0, type=int,
                    help='Seed to use for random values')
parser.add_argument('--device', default=0, type=int,
                    help='GPU ID to train on')
args = parser.parse_args()

args.pin_mem = not args.no_pin_mem

starttime = time.time()

torch.manual_seed(args.seed)

if not args.checkpoint_dir:
    args.checkpoint_dir = f"data/checkpoints/{args.name}-{int(starttime)}/"

if not os.path.exists(args.checkpoint_dir):
    os.mkdir(args.checkpoint_dir)

print("Importing Arrays")
if not os.path.exists(f"data/arrays/X_train_{args.img_size}.npy"):
    print("Arrays not found, generating...")
    preprocessor = PreprocessImages("F:/Datasets/NIH X-Rays/data",
                                    args.img_size)
    (X_train, y_train), (X_test, y_test) = preprocessor()

else:
    X_train = np.load(open(f"data/arrays/X_train_{args.img_size}.npy", "rb"))
    y_train = np.load(open(f"data/arrays/y_train_{args.img_size}.npy", "rb"))
    X_test = np.load(open(f"data/arrays/X_test_{args.img_size}.npy", "rb"))
    y_test = np.load(open(f"data/arrays/y_test_{args.img_size}.npy", "rb"))

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

model.to(args.device)
if args.checkpoint:
    model.load_state_dict(torch.load(args.checkpoint))

loss_fn = nn.MultiLabelSoftMarginLoss().to(args.device)
optimizer = SGD(model.parameters(), lr=1e-6, momentum=0.9)
scheduler = lr_scheduler.ReduceLROnPlateau(optimizer)
mse_fn = nn.MSELoss()

X_train = torch.Tensor(X_train)
y_train = torch.Tensor(y_train)
X_test = torch.Tensor(X_test)
y_test = torch.Tensor(y_test)

dataset = TensorDataset(X_train, y_train)
train_set, val_set = random_split(dataset, [70000, 16524])
test_set = TensorDataset(X_test, y_test)
traindata = DataLoader(train_set, shuffle=True,
                       pin_memory=args.pin_mem, batch_size=args.batch_size)
valdata = DataLoader(val_set, shuffle=True,
                     pin_memory=args.pin_mem, batch_size=args.batch_size)
testdata = DataLoader(test_set, shuffle=True,
                      pin_memory=args.pin_mem, batch_size=args.batch_size)

best_model_wts = copy.deepcopy(model.state_dict())

writer = SummaryWriter(f"data/logs/{args.name}-{int(starttime)}")
dummy_input_shape = (1, 1, args.img_size, args.img_size)
dummy_input = torch.randn(*dummy_input_shape, device=args.device)
writer.add_graph(model, dummy_input)
writer.flush()
scaler = GradScaler()
for epoch in range(args.epochs):
    print(f"Epoch {epoch+1}/{args.epochs}")
    print('='*61)

    running_loss = 0.0
    running_mse = 0.0
    print('Training')
    progressbar = tqdm(traindata, unit='steps', dynamic_ncols=True)
    for index, (inputs, labels) in enumerate(progressbar):

        inputs, labels = inputs.to(args.device), labels.to(args.device)

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
        progressbar.set_description(f'Loss: {running_loss/(index+1):.5f}, \
MSE: {running_mse/(index+1):.5f}')
        progressbar.refresh()

    epoch_loss = running_loss / len(traindata)
    epoch_mse = running_mse / len(traindata)

    running_loss = 0.0
    running_mse = 0.0

    model.eval()
    print('\nValidation')
    progressbar = tqdm(valdata, unit='steps', dynamic_ncols=True)
    for index, (inputs, labels) in enumerate(progressbar):

        inputs, labels = inputs.to(args.device), labels.to(args.device)

        with torch.no_grad():
            with autocast():
                outputs = model(inputs)
                loss = loss_fn(outputs, labels)
                mse = mse_fn(outputs, labels)

        running_loss += loss.item()
        running_mse += mse.item()
        progressbar.set_description(f'Val loss: {running_loss/(index+1):.5f}, \
Val MSE: {running_mse/(index+1):.5f}')
        progressbar.refresh()

    val_loss = running_loss / len(valdata)
    val_mse = running_mse / len(valdata)
    scheduler.step(val_mse)

    if epoch == 0:
        best_loss = val_loss
        best_model_wts = copy.deepcopy(model.state_dict())
        path = os.path.join(args.checkpoint_dir, 'best_weights.pth')
        torch.save(best_model_wts, path)
    elif val_loss < best_loss:
        best_loss = val_loss
        best_model_wts = copy.deepcopy(model.state_dict())
        path = os.path.join(args.checkpoint_dir, 'best_weights.pth')
        torch.save(best_model_wts, path)

    loss_data = {'Training': epoch_loss, 'Validation': val_loss}
    mse_data = {'Training': epoch_mse, 'Validation': val_mse}
    writer.add_scalars('Loss', loss_data, epoch+1)
    writer.add_scalars('MSE', mse_data, epoch+1)
    writer.flush()

    checkpoint_path = os.path.join(args.checkpoint_dir,
                                   f"checkpoint-{epoch:03d}.pth")
    torch.save(model.state_dict(), checkpoint_path)
    print(f"Checkpoint saved to checkpoint-{epoch:03d}.pth\n")

writer.close()

time_elapsed = time.time() - starttime
print(f"Training complete in {time_elapsed // 3600}h \
{time_elapsed // 60}m {round(time_elapsed % 60)}s")

model.load_state_dict(best_model_wts)
model.eval()
running_loss = 0.0
running_mse = 0.0

print('Testing')
progressbar = tqdm(testdata, dynamic_ncols=True)
for index, (inputs, labels) in enumerate(progressbar):

    inputs, labels = inputs.to(args.device), labels.to(args.device)

    with torch.no_grad():
        with autocast():
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)
            mse = mse_fn(outputs, labels)

    running_loss += loss.item()
    running_mse += mse.item()
    progressbar.set_description(f'Test loss: {running_loss/(index+1):.5f}, \
Test MSE: {running_mse/(index+1):.5f}')
    progressbar.refresh()

print("Saving model weights")
savepath = f"data/models/{args.name}-{int(starttime)}.pth"
savepath_weights = f"data/models/{args.name}-{int(starttime)}_weights.pth"
torch.save(model.state_dict(), savepath_weights)
torch.save(model, savepath)
print("Model saved!\n")
