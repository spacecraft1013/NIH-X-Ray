import numpy as np
import os
from multithreaded_preprocessing import PreprocessImages
import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
import time
import copy
import ray

model_save_name = "densenet201_pytorch"
epochs = 250
image_size = 128

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
model.cuda()

loss_fn = nn.MultiLabelMarginLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-6, momentum=0.9)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, verbose=True)

X_train = torch.Tensor(X_train).cuda()
y_train = torch.Tensor(y_train).cuda()
X_test = torch.Tensor(X_test).cuda()
y_test = torch.Tensor(y_test).cuda()

traindataset = TensorDataset(X_train, y_train)
traindata = DataLoader(traindataset, shuffle=True)
valdataset = TensorDataset(X_test, y_test)
valdata = DataLoader(valdataset, shuffle=True)

dataloader = {'Train': traindata, 'Val': valdata}

starttime = time.time()

best_model_wts = copy.deepcopy(model.state_dict())
best_acc = 0.0

scaler = GradScaler()

for epoch in range(epochs):
    print(f"Epoch {epoch+1}/{epochs}")
    print('='*50)

    for phase in ['Train', 'Val']:
        if phase == 'Train':
            model.train()
        else:
            model.eval()

        running_loss = 0.0
        running_corrects = 0

        for index, data in enumerate(dataloader[phase]):
            inputs, labels = data
            print(f"{index}/{len(dataloader[phase])}", end=' ')

            optimizer.zero_grad()

            with torch.set_grad_enabled(phase == 'Train'):
                with autocast():
                    outputs = model(inputs)
                    loss = loss_fn(outputs, labels.long())

                if phase == 'Train':
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
            
            running_loss += loss.item() * inputs.size(0)
            running_corrects += sum(1 for x in zip(outputs[0], labels[0]) if x[0] == x[1])

            print(f"Accuracy: {running_corrects/(index/15)}% Loss: {running_loss/index}", end='\r')

        if phase == 'Train':
            scheduler.step(loss)

        epoch_loss = running_loss / len(dataloader[phase])
        epoch_acc = running_corrects / len(dataloader[phase])
        print(f'{phase}\nLoss: {epoch_loss}\nAccuracy: {epoch_acc}')

        if phase == 'Val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

time_elapsed = time.time() - starttime
print(f"Training complete in {time_elapsed // 60}m {time_elapsed % 60}s")
print('\nBest val Acc: {:4f}'.format(best_acc))
model.load_state_dict(best_model_wts)
print("Saving model")
torch.save(model.state_dict(), f"data/models/{model_save_name}_weights.pth")