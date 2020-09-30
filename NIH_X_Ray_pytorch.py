import numpy as np
import os
from multithreaded_preprocessing import PreprocessImages
import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
import time
import copy

model_save_name = "densenet201_pytorch"
epochs = 250
image_size = 256
batch_size = 32
checkpoint_dir = f"data/checkpoints/{model_save_name}/{time.asctime(time.localtime(time.time())).replace(' ', '_')}"

if os.path.exists(f"data/arrays/X_train_{image_size}.npy") == False:
    preprocessor = PreprocessImages("F:/Datasets/NIH X-Rays/data", image_size=image_size)
    preprocessor.start()

if torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    print("GPU not found, using CPU")
    device = torch.device('cpu')

print("Importing Arrays")
X_train = np.load(open(f"data/arrays/X_train_{image_size}.npy", "rb"))
y_train = np.load(open(f"data/arrays/y_train_{image_size}.npy", "rb"))
X_test = np.load(open(f"data/arrays/X_test_{image_size}.npy", "rb"))
y_test = np.load(open(f"data/arrays/y_test_{image_size}.npy", "rb"))

X_train = np.transpose(X_train, (0, 3, 1, 2))
X_test = np.transpose(X_test, (0, 3, 1, 2))

model = torch.hub.load('pytorch/vision:v0.6.0', 'densenet201', pretrained=False)
model.features[0] = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
model.classifier = nn.Linear(in_features=1920, out_features=15, bias=True)

# model = model_pytorch.Model()
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
test_set = TensorDataset(X_test, y_test)
traindata = DataLoader(train_set, shuffle=True, pin_memory=True, batch_size=batch_size)
valdata = DataLoader(val_set, shuffle=True, pin_memory=True, batch_size=batch_size)
testdata = DataLoader(test_set, shuffle=True, pin_memory=True, batch_size=batch_size)

starttime = time.time()

best_model_wts = copy.deepcopy(model.state_dict())
best_acc = 0.0

scaler = GradScaler()
for epoch in range(epochs+1):
    print(f"Epoch {epoch+1}/{epochs}")
    print('='*61)

    running_loss = 0.0
    running_corrects = 0

    for index, data in enumerate(traindata):
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
        listcorrect = []
        for x in zip(outputs[0], labels[0]):
            if round(float(x[0])) == x[1]:
                listcorrect.append(1)
            else:
                listcorrect.append(0)

        if sum(listcorrect) == 15:
            running_corrects += 1

        print(f"Correct: {running_corrects} Accuracy: {running_corrects/(index+1)/15:.2f}% Loss: {running_loss/(index+1)}", end='\r')

        scheduler.step(loss)

    epoch_loss = running_loss / len(traindata)
    epoch_acc = running_corrects / len(traindata)
    print(f'\nTraining\nLoss: {epoch_loss}\nAccuracy: {epoch_acc}')

    running_loss = 0.0
    running_corrects = 0

    model.eval()
    for index, data in enumerate(valdata):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        print(f"{index+1}/{len(valdata)}", end=' ')

        with torch.no_grad():
            with autocast():
                outputs = model(inputs)
                loss = loss_fn(outputs, labels.long())

        running_loss += loss.item() * inputs.size(0)
        listcorrect = []
        for x in zip(outputs[0], labels[0]):
            if round(float(x[0])) == x[1]:
                listcorrect.append(1)
            else:
                listcorrect.append(0)

        if sum(listcorrect) == 15:
            running_corrects += 1

        print(f"Correct: {running_corrects} Accuracy: {running_corrects/(index+1)/15:.2f}% Loss: {running_loss/(index+1)}", end='\r')

    val_loss = running_loss / len(valdata)
    val_acc = running_corrects / len(valdata)
    print(f'\nValidation\nLoss: {val_loss}\nAccuracy: {val_acc}')

    if val_acc > best_acc:
        best_acc = val_acc
        best_model_wts = copy.deepcopy(model.state_dict())

    torch.save(model.state_dict(), os.path.join(checkpoint_dir, f"checkpoint-{epoch:03d}.pth"))
    print(f"Checkpoint saved to checkpoint-{epoch:03d}.pth")

time_elapsed = time.time() - starttime
print(f"Training complete in {time_elapsed // 60}m {time_elapsed % 60}s")

model.eval()
running_loss = 0.0
running_corrects = 0
for index, data in enumerate(testdata):
    inputs, labels = data
    inputs, labels = inputs.to(device), labels.to(device)
    print(f"{index+1}/{len(valdata)}", end=' ')
    with torch.no_grad():
        with autocast():
            outputs = model(inputs)
            loss = loss_fn(outputs, labels.long())
    
    running_loss += loss.item() * inputs.size(0)
    listcorrect = []
    for x in zip(outputs[0], labels[0]):
        if round(float(x[0])) == x[1]:
            listcorrect.append(1)
        else:
            listcorrect.append(0)
    if sum(listcorrect) == 15:
        running_corrects += 1
    print(f"Correct: {running_corrects} Accuracy: {running_corrects/(index+1)/15:.2f}% Loss: {running_loss/(index+1)}", end='\r')
test_loss = running_loss / len(testdata)
test_acc = running_corrects / len(testdata)
print(f'\nTesting\nLoss: {test_loss}\nAccuracy: {test_acc}')

print("Saving model")
torch.save(model.state_dict(), f"data/models/{model_save_name}_weights.pth")