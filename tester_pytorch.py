import torch
import torch.nn as nn
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import numpy as np

IMAGE_SIZE = 256
BATCH_SIZE = 32

if torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    print("GPU not found, using CPU")
    device = torch.device('cpu')

X_test = np.load(open(f"data/arrays/X_test_{IMAGE_SIZE}.npy", "rb"))
y_test = np.load(open(f"data/arrays/y_test_{IMAGE_SIZE}.npy", "rb"))
X_test = np.transpose(X_test, (0, 3, 1, 2))

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

model.to(device)

loss_fn = nn.MultiLabelMarginLoss().to(device)

test_set = TensorDataset(X_test, y_test)
testdata = DataLoader(test_set, shuffle=True,
                      pin_memory=True, batch_size=BATCH_SIZE)

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
