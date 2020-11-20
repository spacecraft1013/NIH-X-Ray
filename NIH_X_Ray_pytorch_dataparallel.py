import copy
import os
import time

import numpy as np
import onnx
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.nn import DataParallel as DP
from torch.optim import SGD, lr_scheduler
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from multithreaded_preprocessing import PreprocessImages

MODEL_SAVE_NAME = "densenet201_pytorch"
EPOCHS = 250
IMAGE_SIZE = 256
BATCH_SIZE = 32
CHECKPOINT_DIR = f"data/checkpoints/{MODEL_SAVE_NAME}/"

if not os.path.exists(CHECKPOINT_DIR):
    os.mkdir(CHECKPOINT_DIR)

if torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    print("GPU not found, using CPU")
    device = torch.device('cpu')

print("Importing Arrays")
if not os.path.exists(f"data/arrays/X_train_{IMAGE_SIZE}.npy"):
    print("Arrays not found, generating...")
    preprocessor = PreprocessImages("F:/Datasets/NIH X-Rays/data", IMAGE_SIZE)
    (X_train, y_train), (X_test, y_test) = preprocessor()

else:
    X_train = np.load(open(f"data/arrays/X_train_{IMAGE_SIZE}.npy", "rb"))
    y_train = np.load(open(f"data/arrays/y_train_{IMAGE_SIZE}.npy", "rb"))
    X_test = np.load(open(f"data/arrays/X_test_{IMAGE_SIZE}.npy", "rb"))
    y_test = np.load(open(f"data/arrays/y_test_{IMAGE_SIZE}.npy", "rb"))

# Convert channels-last to channels-first format
X_train = np.transpose(X_train, (0, 3, 1, 2))
X_test = np.transpose(X_test, (0, 3, 1, 2))

model = torch.hub.load('pytorch/vision:v0.6.0', 'densenet201',
                       pretrained=False)
model.features[0] = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2),
                              padding=(3, 3), bias=False)
model.classifier = nn.Linear(in_features=1920, out_features=15, bias=True)

model = DP(model)

loss_fn = nn.MultiLabelMarginLoss().to(device)
optimizer = SGD(model.parameters(), lr=1e-6, momentum=0.9)
scheduler = lr_scheduler.ReduceLROnPlateau(optimizer)

X_train = torch.Tensor(X_train)
y_train = torch.Tensor(y_train)
X_test = torch.Tensor(X_test)
y_test = torch.Tensor(y_test)

dataset = TensorDataset(X_train, y_train)
train_set, val_set = torch.utils.data.random_split(dataset, [70000, 16524])
val_labels = [label.numpy() for input, label in val_set]
test_set = TensorDataset(X_test, y_test)
traindata = DataLoader(train_set, shuffle=True,
                       pin_memory=True, batch_size=BATCH_SIZE)
valdata = DataLoader(val_set, shuffle=True,
                     pin_memory=True, batch_size=BATCH_SIZE)
testdata = DataLoader(test_set, shuffle=True,
                      pin_memory=True, batch_size=BATCH_SIZE)

starttime = time.time()

best_model_wts = copy.deepcopy(model.state_dict())

scaler = GradScaler()
for epoch in range(EPOCHS+1):
    print(f"Epoch {epoch+1}/{EPOCHS}")
    print('='*61)

    running_loss = 0.0
    print('Training')
    progressbar = tqdm(traindata, unit='steps', dynamic_ncols=True)
    for index, (inputs, labels) in enumerate(progressbar):

        inputs, labels = inputs.to(device), labels.to(device)

        model.train()
        optimizer.zero_grad()

        with torch.set_grad_enabled(True):
            with autocast():
                outputs = model(inputs)
                loss = loss_fn(outputs, labels.long())
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        running_loss += loss.item() * inputs.size(0)
        progressbar.set_description(f'Loss: {running_loss/(index+1):.5f}')
        progressbar.refresh()

        scheduler.step(loss)

    epoch_loss = running_loss / len(traindata)

    running_loss = 0.0

    model.eval()
    print('Validation')
    progressbar = tqdm(valdata, unit='steps', dynamic_ncols=True)
    for index, (inputs, labels) in enumerate(progressbar):

        inputs, labels = inputs.to(device), labels.to(device)

        with torch.no_grad():
            with autocast():
                outputs = model(inputs)
                loss = loss_fn(outputs, labels.long())

        running_loss += loss.item() * inputs.size(0)
        progressbar.set_description(f'Val loss: {running_loss/(index+1):.5f}')
        progressbar.refresh()

    val_loss = running_loss / len(valdata)

    if epoch == 0:
        best_loss = val_loss
        best_model_wts = copy.deepcopy(model.state_dict())
    elif val_loss < best_loss:
        best_loss = val_loss
        best_model_wts = copy.deepcopy(model.state_dict())

    checkpoint_path = os.path.join(CHECKPOINT_DIR,
                                   f"checkpoint-{epoch:03d}.pth")
    torch.save(model.state_dict(), checkpoint_path)
    print(f"Checkpoint saved to checkpoint-{epoch:03d}.pth\n")

time_elapsed = time.time() - starttime
print(f"Training complete in {time_elapsed // 60}m {time_elapsed % 60}s")

model.load_state_dict(best_model_wts)
model.eval()
running_loss = 0.0

print('Testing')
progressbar = tqdm(testdata, dynamic_ncols=True)
for index, (inputs, labels) in enumerate(progressbar):

    inputs, labels = inputs.to(device), labels.to(device)

    with torch.no_grad():
        with autocast():
            outputs = model(inputs)
            loss = loss_fn(outputs, labels.long())

    running_loss += loss.item() * inputs.size(0)
    progressbar.set_description(f'Test loss: {running_loss/(index+1):.5f}')
    progressbar.refresh()

print("Saving model weights")
torch.save(model.state_dict(), f"data/models/{MODEL_SAVE_NAME}_weights.pth")
print("Model saved!\n")

print("Saving ONNX file")
onnx_save_path = f"data/models/{MODEL_SAVE_NAME}.onnx"
dummy_input = torch.randn(1, 1, IMAGE_SIZE, IMAGE_SIZE, device='cuda')
torch.onnx.export(model, dummy_input, onnx_save_path, verbose=True)
onnx.checker.check_model(onnx_save_path)
print("ONNX model has been successfully saved!")
