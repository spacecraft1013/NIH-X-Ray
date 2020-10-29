import copy
import os
import time

import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, TensorDataset

from multithreaded_preprocessing import PreprocessImages

model_save_name = "densenet201_pytorch"
current_time = time.asctime(time.localtime(time.time())).replace(' ', '_').replace(':', '_')
epochs = 250
image_size = 256
batch_size = 32
checkpoint_dir = f"data/checkpoints/{model_save_name}/{current_time}/"

try:
    os.mkdir(checkpoint_dir)
except OSError:
    os.mkdir(f"data/checkpoints/{model_save_name}")
    os.mkdir(checkpoint_dir)


if torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    print("GPU not found, using CPU")
    device = torch.device('cpu')

print("Importing Arrays")
if not os.path.exists(f"data/arrays/X_train_{image_size}.npy"):
    print("Arrays not found, generating...")
    preprocessor = PreprocessImages("F:/Datasets/NIH X-Rays/data", image_size)
    (X_train, y_train), (X_test, y_test) = preprocessor()

else:
    X_train = np.load(open(f"data/arrays/X_train_{image_size}.npy", "rb"))
    y_train = np.load(open(f"data/arrays/y_train_{image_size}.npy", "rb"))
    X_test = np.load(open(f"data/arrays/X_test_{image_size}.npy", "rb"))
    y_test = np.load(open(f"data/arrays/y_test_{image_size}.npy", "rb"))

# Convert channels-last to channels-first format
X_train = np.transpose(X_train, (0, 3, 1, 2))
X_test = np.transpose(X_test, (0, 3, 1, 2))

model = torch.hub.load('pytorch/vision:v0.6.0', 'densenet201', pretrained=False)
model.features[0] = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
model.classifier = nn.Linear(in_features=1920, out_features=15, bias=True)

model.to(device)

loss_fn = nn.MultiLabelMarginLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-6, momentum=0.9)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, verbose=True)

X_train = torch.Tensor(X_train)
y_train = torch.Tensor(y_train)
X_test = torch.Tensor(X_test)
y_test = torch.Tensor(y_test)

dataset = TensorDataset(X_train, y_train)
train_set, val_set = torch.utils.data.random_split(dataset, [70000, 16524])
val_labels = [label.numpy() for input, label in val_set]
test_set = TensorDataset(X_test, y_test)
traindata = DataLoader(train_set, shuffle=True, pin_memory=True, batch_size=batch_size)
valdata = DataLoader(val_set, shuffle=True, pin_memory=True, batch_size=batch_size)
testdata = DataLoader(test_set, shuffle=True, pin_memory=True, batch_size=batch_size)

starttime = time.time()

best_model_wts = copy.deepcopy(model.state_dict())

scaler = GradScaler()
for epoch in range(epochs+1):
    print(f"Epoch {epoch+1}/{epochs}")
    print('='*61)

    running_loss = 0.0

    for index, data in enumerate(traindata):
        startstep = time.time()
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        print(f"{index+1}/{len(traindata)}", end=' ')

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

        print(f"Training: {(time.time() - startstep)*1000:.2f}ms/step, Loss: {running_loss/(index+1)}", end='\r')

        scheduler.step(loss)
    print()

    epoch_loss = running_loss / len(traindata)

    running_loss = 0.0

    model.eval()
    y_pred = []
    for index, data in enumerate(valdata):
        startstep = time.time()
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        print(f"{index+1}/{len(valdata)}", end=' ')

        with torch.no_grad():
            with autocast():
                outputs = model(inputs)
                loss = loss_fn(outputs, labels.long())

        y_pred.extend(outputs.cpu().numpy())

        running_loss += loss.item() * inputs.size(0)

        print(f"Validation: {(time.time() - startstep)*1000:.2f}ms/step, Loss: {running_loss/(index+1)}", end='\r')
    print()

    val_loss = running_loss / len(valdata)

    if epoch == 0:
        best_loss = val_loss
        best_model_wts = copy.deepcopy(model.state_dict())
    elif val_loss < best_loss:
        best_loss = val_loss
        best_model_wts = copy.deepcopy(model.state_dict())

    torch.save(model.state_dict(), os.path.join(checkpoint_dir, f"checkpoint-{epoch:03d}.pth"))
    print(f"Checkpoint saved to checkpoint-{epoch:03d}.pth")

time_elapsed = time.time() - starttime
print(f"Training complete in {time_elapsed // 60}m {time_elapsed % 60}s")

model.eval()
running_loss = 0.0
y_pred = []
for index, data in enumerate(testdata):
    startstep = time.time()
    inputs, labels = data
    inputs, labels = inputs.to(device), labels.to(device)
    print(f"{index+1}/{len(valdata)}", end=' ')

    with torch.no_grad():
        with autocast():
            outputs = model(inputs)
            loss = loss_fn(outputs, labels.long())

    y_pred.extend(outputs.cpu().numpy())

    running_loss += loss.item() * inputs.size(0)

    print(f"Testing: {(time.time() - startstep)*1000:.2f}ms/step, Loss: {running_loss/(index+1)}", end='\r')
print()

test_loss = running_loss / len(testdata)

print("Saving model")
torch.save(model.state_dict(), f"data/models/{model_save_name}_weights.pth")