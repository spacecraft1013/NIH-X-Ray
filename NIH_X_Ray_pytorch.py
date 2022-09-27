import argparse
import copy
import datetime
import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, TensorDataset, random_split
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from multithreaded_preprocessing import PreprocessImages
from utils import Config

# Parse command line arguments for the program

parser = argparse.ArgumentParser()
parser.add_argument('--resume-latest', default=False, action='store_true',
                    help='Resume from latest checkpoint')
parser.add_argument('-c', '--checkpoint', default=None, type=str,
                    help='Checkpoint file to load from')
parser.add_argument('--name', default='model', type=str,
                    help='Name to save model (no file extension)')

config = Config("model_config.yml")
parser.parse_args(namespace=config)
starting_epoch = None

# Set original timestamp for logging/checkpoint directories
starttime = datetime.datetime.now()

# Set the initial seed for training
torch.manual_seed(config.seed)

torch.backends.cudnn.benchmark = config.cudnn_benchmark

# Create the checkpoint and logging directories
config.checkpoint_dir = os.path.join(config.checkpoint_dir, f"{config.name}-{int(starttime.timestamp())}")
config.log_dir = os.path.join(config.log_dir, f"{config.name}-{int(starttime.timestamp())}")

# Find latest checkpoint/logging directory and continue training from there
if config.resume_latest:
    checkpoint_dirs = os.listdir(config.checkpoint_dir)
    checkpoint_dirs = [i for i in checkpoint_dirs if i[-10:].isdigit()]
    timestamps = [int(i[-10:]) for i in checkpoint_dirs]
    max_index = timestamps.index(max(timestamps))
    config.checkpoint_dir = os.path.join(config.checkpoint_dir, checkpoint_dirs[max_index])
    checkpoints = os.listdir(config.checkpoint_dir)
    checkpoint_nums = []
    for checkpoint in checkpoints:
        temp = []
        for i in checkpoint:
            if i.isdigit():
                temp.append(i)
        if temp:
            checkpoint_nums.append(int(''.join(temp)))
    max_index = checkpoint_nums.index(max(checkpoint_nums))
    starting_epoch = max(checkpoint_nums)+1
    config.checkpoint = os.path.join(config.checkpoint_dir, checkpoints[max_index])

    log_dirs = os.listdir(config.log_dir)
    log_dirs = [i for i in log_dirs if i[-10:].isdigit()]
    timestamps = [int(i[-10:]) for i in log_dirs]
    max_index = timestamps.index(max(timestamps))
    config.log_dir = os.path.join(config.log_dir, log_dirs[max_index])

# Create checkpoint directory folder
if not os.path.exists(config.checkpoint_dir):
    os.makedirs(config.checkpoint_dir, exist_ok=True)
if not os.path.exists(config.log_dir):
    os.makedirs(config.log_dir, exist_ok=True)

print("Importing Arrays")
# If array files are not found, preprocess images to create arrays
if not os.path.exists(os.path.join(config.array_dir, f"arrays_{config.input_size[0]}.npz")):
    print("Arrays not found, generating...")
    preprocessor = PreprocessImages(config.dataset_dir,
                                    config.input_size[0])
    X_train, y_train, X_test, y_test = preprocessor()

# Import preprocessed data arrays
else:
    arrays = np.load(os.path.join(config.array_dir, f"arrays_{config.input_size[0]}.npz"))
    X_train = arrays['X_train']
    y_train = arrays['y_train']
    X_test = arrays['X_test']
    y_test = arrays['y_test']

# Convert channels-last to channels-first format
X_train = np.transpose(X_train, (0, 3, 1, 2))
X_test = np.transpose(X_test, (0, 3, 1, 2))

# Load densenet201 model and modify to work for these images
model = torch.hub.load('pytorch/vision:v0.11.1', 'densenet201',
                       pretrained=True)
model.features[0] = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2),
                              padding=(3, 3), bias=False)
model.classifier = nn.Sequential(
    nn.Linear(in_features=1920, out_features=14, bias=True),
    nn.Sigmoid()
)

# Move model to GPU memory
model.to(config.device_config.devices[0])
# Load model checkpoint
if config.checkpoint:
    model.load_state_dict(torch.load(config.checkpoint))

# Initialize loss and optimizer functions
if config.loss_algorithm == 'cross_entropy':
    loss_alg = nn.CrossEntropyLoss
elif config.loss_algorithm == 'binary_crossentropy':
    loss_alg = nn.BCEWithLogitsLoss
elif config.loss_algorithm == 'mean_squared_error':
    loss_alg = nn.MSELoss
else:
    raise ValueError("Invalid loss algorithm")

if config.optimizer == 'adamax':
    optimizer = optim.Adamax
elif config.optimizer == 'adam':
    optimizer = optim.Adam
else:
    raise ValueError("Invalid optimizer")

loss_fn = loss_alg().to(config.device_config.devices[0])
optimizer = optimizer(model.parameters(), lr=config.learning_rate)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer)
mse_fn = nn.MSELoss()

# Convert NumPy arrays to PyTorch tensors
X_train = torch.Tensor(X_train)
y_train = torch.Tensor(y_train)
X_test = torch.Tensor(X_test)
y_test = torch.Tensor(y_test)

# Create PyTorch dataset and initialize dataloaders
dataset = TensorDataset(X_train, y_train)
train_num = int(len(dataset)*0.7)
train_set, val_set = random_split(dataset, [train_num, len(dataset)-train_num])
test_set = TensorDataset(X_test, y_test)
traindata = DataLoader(train_set, shuffle=True, drop_last=True,
                       pin_memory=config.pin_mem, batch_size=config.batch_size)
valdata = DataLoader(val_set, shuffle=True, drop_last=True,
                     pin_memory=config.pin_mem, batch_size=config.batch_size)
testdata = DataLoader(test_set, shuffle=True, drop_last=True,
                      pin_memory=config.pin_mem, batch_size=config.batch_size)

best_model_wts = copy.deepcopy(model.state_dict())

# Initialize tensorboard logging
writer = SummaryWriter(config.log_dir)

# Create model graph in tensorboard
dummy_input_shape = (1, 1, *config.input_size)
dummy_input = torch.randn(*dummy_input_shape, device=config.device_config.devices[0])
writer.add_graph(model, dummy_input)
writer.flush()

# Initialize gradient scaler
scaler = GradScaler()

# Iterate over epochs
for epoch in range(config.epochs):
    if starting_epoch:
        epoch += starting_epoch
    print(f"Epoch {epoch+1}/{config.epochs}")
    print('='*61)

    # Initialize running loss and running mse
    running_loss = 0.0
    running_mse = 0.0

    # Set model to training mode
    model.train()

    # Create progress bar
    progressbar = tqdm(traindata, desc='Training', unit='steps', dynamic_ncols=True)

    # Iterate over dataloader
    for index, (inputs, labels) in enumerate(progressbar):

        # Move data to GPU memory
        inputs, labels = inputs.to(config.device_config.devices[0]), labels.to(config.device_config.devices[0])

        # Zero gradient for optimizer
        optimizer.zero_grad(set_to_none=True)

        # Enable gradient
        with torch.set_grad_enabled(True):

            # Autocast the model with automatic mixed precision
            with autocast():
                outputs = model(inputs)
                loss = loss_fn(outputs, labels)
                mse = mse_fn(outputs, labels)

            # Use scaler to scale gradients
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        # Update running loss and mse
        running_loss += loss.item()
        running_mse += mse.item()
        progressbar.set_description(f'Training, Loss: {running_loss/(index+1):.5f}, \
MSE: {running_mse/(index+1):.5f}')
        progressbar.refresh()

    epoch_loss = running_loss / len(traindata)
    epoch_mse = running_mse / len(traindata)

    # Reset running loss and mse
    running_loss = 0.0
    running_mse = 0.0

    # Set model to evaluation mode
    model.eval()

    # Create progress bar
    progressbar = tqdm(valdata, desc="Validation", unit='steps', dynamic_ncols=True)

    # Iterate over dataloader
    for index, (inputs, labels) in enumerate(progressbar):

        # Move data to GPU memory
        inputs, labels = inputs.to(config.device_config.devices[0]), labels.to(config.device_config.devices[0])

        # Disable gradient
        with torch.no_grad():

            # Autocast the model with automatic mixed precision
            with autocast():
                outputs = model(inputs)
                loss = loss_fn(outputs, labels)
                mse = mse_fn(outputs, labels)

        # Update running loss and mse
        running_loss += loss.item()
        running_mse += mse.item()
        progressbar.set_description(f'Validation, Loss: {running_loss/(index+1):.5f}, \
MSE: {running_mse/(index+1):.5f}')
        progressbar.refresh()

    val_loss = running_loss / len(valdata)
    val_mse = running_mse / len(valdata)
    scheduler.step(val_mse)

    # Initialize best loss if the variable does not exist
    if 'best_loss' not in locals() or val_mse < best_loss:
        best_loss = val_mse
        best_model_wts = copy.deepcopy(model.state_dict())
        path = os.path.join(config.checkpoint_dir, 'best_weights.pth')
        torch.save(best_model_wts, path)

    # Log the loss and mse to tensorboard
    loss_data = {'Training': epoch_loss, 'Validation': val_loss}
    mse_data = {'Training': epoch_mse, 'Validation': val_mse}
    writer.add_scalars('Loss', loss_data, epoch+1)
    writer.add_scalars('MSE', mse_data, epoch+1)
    writer.flush()

    # Save model checkpoint
    checkpoint_path = os.path.join(config.checkpoint_dir,
                                   f"checkpoint-{epoch:03d}.pth")
    torch.save(model.state_dict(), checkpoint_path)
    print(f"Checkpoint saved to checkpoint-{epoch:03d}.pth\n")

# Close the tensorboard logger
writer.close()

endtime = datetime.datetime.now()
print(f"Training complete in {endtime - starttime}")

# Load best weights
model.load_state_dict(best_model_wts)

# Set the model to evalutation mode
model.eval()

# Reset running loss and mse
running_loss = 0.0
running_mse = 0.0

print('Testing')

# Create progress bar
progressbar = tqdm(testdata, desc="Testing", unit='steps', dynamic_ncols=True)

# Iterate over dataloader
for index, (inputs, labels) in enumerate(progressbar):

    # Move data to GPU memory
    inputs, labels = inputs.to(config.device_config.devices[0]), labels.to(config.device_config.devices[0])

    # Disable gradient
    with torch.no_grad():

        # Autocast the model with automatic mixed precision
        with autocast():
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)
            mse = mse_fn(outputs, labels)

    # Update running loss and mse
    running_loss += loss.item()
    running_mse += mse.item()
    progressbar.set_description(f'Test loss: {running_loss/(index+1):.5f}, \
Test MSE: {running_mse/(index+1):.5f}')
    progressbar.refresh()

# Save model weights
print("Saving model weights")
savepath = os.path.join(config.model_dir, f"{config.name}-{int(starttime.timestamp())}_weights.pth")
torch.save(model.state_dict(), savepath)
print("Model saved!\n")
