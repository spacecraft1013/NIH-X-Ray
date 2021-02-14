import argparse
import copy
import datetime
import os
import time

import numpy as np
import torch
import torch.nn as nn
from sklearn.utils.class_weight import compute_sample_weight
from torch.cuda.amp import GradScaler, autocast
from torch.optim import Adamax, lr_scheduler
from torch.utils.data import DataLoader, TensorDataset, random_split
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from multithreaded_preprocessing import PreprocessImages

parser = argparse.ArgumentParser()
parser.add_argument('class_name', type=str,
                    help='Class to train on')
parser.add_argument('--resume-latest', default=False, action='store_true',
                    help='Resume from latest checkpoint')
parser.add_argument('-c', '--checkpoint', default=None, type=str,
                    help='Checkpoint file to load from')
parser.add_argument('-e', '--epochs', default=250, type=int,
                    help='Number of epochs to use')
parser.add_argument('--no-pin-mem', default=False, action='store_true',
                    help="Don't use pinned memory")
parser.add_argument('--name', default=None, type=str,
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
parser.add_argument('--device', default=0, type=int,
                    help='GPU ID to train on')
args = parser.parse_args()

args.pin_mem = not args.no_pin_mem

starttime = time.time()

torch.manual_seed(args.seed)

if not args.name:
    args.name = "binary-" + args.class_name

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
    args.checkpoint = checkpoints[max_index]

    log_dirs = os.listdir("data/logs")
    log_dirs = [i for i in log_dirs if i[-10:].isdigit()]
    timestamps = [int(i[-10:]) for i in log_dirs]
    max_index = timestamps.index(max(timestamps))
    args.log_dir = f"data/logs/{log_dirs[max_index]}/"

if not os.path.exists(args.checkpoint_dir):
    os.mkdir(args.checkpoint_dir)

print("Importing Arrays")
if not os.path.exists(f"data/arrays/X_train_{args.img_size}_{args.class_name.lower()}.npy"):
    print("Arrays not found, generating...")
    preprocessor = PreprocessImages("F:/Datasets/NIH X-Rays/data",
                                    args.img_size)
    (X_train, y_train), (X_test, y_test) = preprocessor.start_single(args.class_name.capitalize())

else:
    X_train = np.load(open(f"data/arrays/X_train_{args.img_size}_{args.class_name.lower()}.npy", "rb"))
    y_train = np.load(open(f"data/arrays/y_train_{args.img_size}_{args.class_name.lower()}.npy", "rb"))
    X_test = np.load(open(f"data/arrays/X_test_{args.img_size}_{args.class_name.lower()}.npy", "rb"))
    y_test = np.load(open(f"data/arrays/y_test_{args.img_size}_{args.class_name.lower()}.npy", "rb"))

# Convert channels-last to channels-first format
X_train = np.transpose(X_train, (0, 3, 1, 2))
X_test = np.transpose(X_test, (0, 3, 1, 2))

model = torch.hub.load('pytorch/vision:v0.6.0', 'densenet201',
                       pretrained=False)
model.features[0] = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2),
                              padding=(3, 3), bias=False)
model.classifier = nn.Sequential(
    nn.Linear(in_features=1920, out_features=1, bias=True),
    # nn.Sigmoid()
)

model.to(args.device)
if args.checkpoint:
    model.load_state_dict(torch.load(args.checkpoint))

loss_fn = nn.BCEWithLogitsLoss().to(args.device)
optimizer = Adamax(model.parameters(), lr=1e-6)
scheduler = lr_scheduler.ReduceLROnPlateau(optimizer)
mse_fn = nn.MSELoss()

X_train = torch.Tensor(X_train)
y_train = torch.Tensor(y_train)
X_test = torch.Tensor(X_test)
y_test = torch.Tensor(y_test)

dataset = TensorDataset(X_train, y_train)
train_num = int(len(dataset)*0.7)
train_set, val_set = random_split(dataset, [train_num, len(dataset)-train_num])
test_set = TensorDataset(X_test, y_test)
traindata = DataLoader(train_set, drop_last=True, sampler=trainsampler,
                       pin_memory=args.pin_mem, batch_size=args.batch_size)
valdata = DataLoader(val_set, shuffle=True, drop_last=True,
                     pin_memory=args.pin_mem, batch_size=args.batch_size)
testdata = DataLoader(test_set, shuffle=True, drop_last=True,
                      pin_memory=args.pin_mem, batch_size=args.batch_size)

best_model_wts = copy.deepcopy(model.state_dict())

writer = SummaryWriter(args.log_dir)
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
    running_correct = 0
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

        outputs = torch.round(torch.sigmoid(outputs))
        running_correct += (outputs == labels).sum().float()/args.batch_size
        progressbar.set_description(f'Loss: {running_loss/(index+1):.5f}, \
MSE: {running_mse/(index+1):.5f}, Accuracy: {running_correct/(index+1):.2%}')
        progressbar.refresh()

    epoch_loss = running_loss / len(traindata)
    epoch_mse = running_mse / len(traindata)
    epoch_accuracy = running_correct / len(traindata)

    running_loss = 0.0
    running_mse = 0.0
    running_correct = 0

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
        outputs = torch.round(torch.sigmoid(outputs))
        running_correct += (outputs == labels).sum().float()/args.batch_size
        progressbar.set_description(f'Val loss: {running_loss/(index+1):.5f}, \
Val MSE: {running_mse/(index+1):.5f}, Accuracy: {running_correct/(index+1):.2%}')
        progressbar.refresh()

    val_loss = running_loss / len(valdata)
    val_mse = running_mse / len(valdata)
    val_accuracy = running_correct / len(traindata)
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

    accuracy_data = {'Training': epoch_accuracy*100, 'Validation': val_accuracy*100}
    loss_data = {'Training': epoch_loss, 'Validation': val_loss}
    mse_data = {'Training': epoch_mse, 'Validation': val_mse}
    writer.add_scalars('Accuracy', accuracy_data, epoch+1)
    writer.add_scalars('Loss', loss_data, epoch+1)
    writer.add_scalars('MSE', mse_data, epoch+1)
    writer.flush()

    checkpoint_path = os.path.join(args.checkpoint_dir,
                                   f"checkpoint-{epoch:03d}.pth")
    torch.save(model.state_dict(), checkpoint_path)
    print(f"Checkpoint saved to checkpoint-{epoch:03d}.pth\n")

writer.close()

time_elapsed = time.time() - starttime
print(f"Training complete in {datetime.timedelta(seconds=time_elapsed)}")

model.load_state_dict(best_model_wts)
model.eval()
running_loss = 0.0
running_mse = 0.0
running_correct = 0

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
    outputs = torch.round(torch.sigmoid(outputs))
    running_correct += (outputs == labels).sum().float()/args.batch_size
    progressbar.set_description(f'Test loss: {running_loss/(index+1):.5f}, \
Test MSE: {running_mse/(index+1):.5f}, Accuracy: {running_correct/(index+1):.2%}')
    progressbar.refresh()

print("Saving model weights")
savepath = f"data/models/{args.name}-{int(starttime)}.pth"
savepath_weights = f"data/models/{args.name}-{int(starttime)}_weights.pth"
torch.save(model.state_dict(), savepath_weights)
torch.save(model, savepath)
print("Model saved!\n")
