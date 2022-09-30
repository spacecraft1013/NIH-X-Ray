import argparse
import copy
import datetime
import os

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn, optim
from torch.cuda.amp import GradScaler, autocast
from torch.distributed.optim import ZeroRedundancyOptimizer
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, TensorDataset, random_split
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from model import NIHXRayModel
from multithreaded_preprocessing import PreprocessImages
from utils import Config


def train(proc: int, scaler: GradScaler, model: nn.Module, starttime: datetime.datetime,
          train_set: Dataset, val_set: Dataset, test_set: Dataset, config: Config):

    rank = config.device_config.node_rank * config.device_config.num_gpus + proc
    dist.init_process_group(
        backend='nccl',
        world_size=config.world_size,
        rank=rank
    )

    torch.backends.cudnn.benchmark = config.cudnn_benchmark
    torch.manual_seed(config.seed)

    if config.device_config.devices:
        gpu_num = config.device_config.devices[proc-(config.device_config.node_rank*config.device_config.num_gpus)]
    else:
        gpu_num = proc-(config.device_config.node_rank*config.device_config.num_gpus)

    model.to(gpu_num)
    ddp_model = DDP(model, device_ids=[gpu_num])

    if config.checkpoint:
        ddp_model.load_state_dict(torch.load(config.checkpoint))

    trainsampler = DistributedSampler(train_set, num_replicas=config.device_config.num_gpus)
    valsampler = DistributedSampler(val_set, num_replicas=config.device_config.num_gpus)
    testsampler = DistributedSampler(test_set, num_replicas=config.device_config.num_gpus)

    traindata = DataLoader(train_set, pin_memory=config.pin_mem, drop_last=True,
                           batch_size=config.batch_size, sampler=trainsampler)
    valdata = DataLoader(val_set, pin_memory=config.pin_mem, drop_last=True,
                         batch_size=config.batch_size, sampler=valsampler)
    testdata = DataLoader(test_set, pin_memory=config.pin_mem, drop_last=True,
                          batch_size=config.batch_size, sampler=testsampler)

    if rank == 0:
        writer = SummaryWriter(config.log_dir)
        size_tuple = (1, 1, *config.img_size)
        dummy_input = torch.randn(*size_tuple, device='cuda:0')
        writer.add_graph(model, dummy_input)
        writer.flush()

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
        optimizer_alg = optim.Adamax
    elif config.optimizer == 'adam':
        optimizer_alg = optim.Adam
    else:
        raise ValueError("Invalid optimizer")

    loss_fn = loss_alg().to(gpu_num)
    optimizer = ZeroRedundancyOptimizer(ddp_model.parameters(), optimizer_class=optimizer_alg, lr=config.learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer)
    mse_fn = nn.MSELoss().to(gpu_num)

    best_model_wts = copy.deepcopy(ddp_model.state_dict())

    for epoch in range(config.epochs):
        if proc == 0:
            print(f"Epoch {epoch+1}/{config.epochs}")
            print('='*61)

        trainsampler.set_epoch(epoch)
        valsampler.set_epoch(epoch)
        testsampler.set_epoch(epoch)

        running_loss = 0.0
        running_mse = 0.0

        ddp_model.train()

        if proc == 0:
            traindata = tqdm(traindata, dynamic_ncols=True, desc="Training", unit_scale=config.world_size)

        for index, (inputs, labels) in enumerate(traindata):

            inputs, labels = inputs.to(gpu_num), labels.to(gpu_num)

            optimizer.zero_grad(set_to_none=True)

            with torch.set_grad_enabled(True):
                with autocast():
                    outputs = ddp_model(inputs)
                    loss = loss_fn(outputs, labels.long())
                    mse = mse_fn(outputs, labels)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

            running_loss += loss.item()
            running_mse += mse.item()

            if gpu_num == 0:
                traindata.set_description(f"Training, Loss: {running_loss/(index+1):.5f}, MSE: {running_mse/(index+1):.5f}")
                traindata.refresh()

        epoch_loss = running_loss / len(traindata)
        epoch_mse = running_mse / len(traindata)
        print()

        running_loss = 0.0
        running_mse = 0.0

        ddp_model.eval()
        if proc == 0:
            valdata = tqdm(valdata, unit='steps', dynamic_ncols=True, desc="Validation", unit_scale=config.world_size)
        for index, (inputs, labels) in enumerate(valdata):

            inputs, labels = inputs.to(gpu_num), labels.to(gpu_num)

            with torch.no_grad():
                with autocast():
                    outputs = ddp_model(inputs)
                    loss = loss_fn(outputs, labels.long())
                    mse = mse_fn(outputs, labels)

            running_loss += loss.item()
            running_mse += mse.item()

            if proc == 0:
                valdata.set_description(f"Validation, Loss: {running_loss/(index+1):.5f}, MSE: {running_mse/(index+1):.5f}")
                valdata.refresh()

        val_loss = running_loss / len(valdata)
        val_mse = running_mse / len(valdata)
        scheduler.step(val_mse)

        if ('best_loss' not in locals() or val_mse < best_loss) and rank == 0:
            best_loss = val_mse
            best_model_wts = copy.deepcopy(model.state_dict())
            path = os.path.join(config.checkpoint_dir, 'best_weights.pth')
            torch.save(best_model_wts, path)

        if rank == 0:
            loss_data = {'Training': epoch_loss, 'Validation': val_loss}
            mse_data = {'Training': epoch_mse, 'Validation': val_mse}
            writer.add_scalars('Loss', loss_data, epoch+1)
            writer.add_scalars('MSE', mse_data, epoch+1)
            writer.flush()

        if rank == 0:
            checkpoint_path = os.path.join(config.checkpoint_dir,
                                        f"checkpoint-{epoch:03d}.pth")
            torch.save(ddp_model.state_dict(), checkpoint_path)
            print(f"Checkpoint saved to checkpoint-{epoch:03d}.pth\n")

        dist.barrier()
        map_location = {'cuda:0': f'cuda:{gpu_num}'}
        ddp_model.load_state_dict(torch.load(checkpoint_path,
                                             map_location=map_location))

    if rank == 0:
        writer.close()
    if proc == 0:
        endtime = datetime.datetime.now()
        print(f"Training complete in {endtime - starttime}")

    ddp_model.load_state_dict(best_model_wts)

    ddp_model.eval()
    running_loss = 0.0
    running_mse = 0.0

    if proc == 0:
        testdata = tqdm(testdata, dynamic_ncols=True, desc="Testing", unit_scale=config.world_size)
    for index, (inputs, labels) in enumerate(testdata):

        inputs, labels = inputs.to(rank), labels.to(rank)

        with torch.no_grad():
            with autocast():
                outputs = ddp_model(inputs)
                loss = loss_fn(outputs, labels)
                mse = mse_fn(outputs, labels)

        running_loss += loss.item()
        running_mse += mse.item()

        if proc == 0:
            testdata.set_description(f"Testing, Loss: {running_loss/(index+1):.5f}, MSE: {running_mse/(index+1):.5f}")
            testdata.refresh()

    if rank == 0:
        print("Saving model weights")
        savepath = os.path.join(config.model_dir, f"{config.name}-{int(starttime.timestamp())}_weights.pth")
        torch.save(ddp_model.state_dict(), savepath)
        print("Model saved!\n")

    dist.destroy_process_group()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume-latest', default=False, action='store_true',
                        help='Resume from latest checkpoint')
    parser.add_argument('-c', '--checkpoint', default=None, type=str,
                        help='Checkpoint file to load from')
    parser.add_argument('--name', default='model', type=str,
                        help='Name to save model (no file extension)')

    config = Config("model_config.yml")
    parser.parse_args(namespace=config)

    config.world_size = config.device_config.num_gpus * config.device_config.num_nodes
    starting_epoch = None

    starttime = datetime.datetime.now()

    mp.set_sharing_strategy('file_system')

    assert config.device_config.num_gpus == len(config.device_config.devices), "Number of GPUs must match number of devices"

    os.environ['MASTER_ADDR'] = config.master_addr
    os.environ['MASTER_PORT'] = config.port

    print("Importing Arrays")
    if not os.path.exists(os.path.join(config.array_dir, f"arrays_{config.input_size[0]}.npz")):
        print("Arrays not found, generating...")
        preprocessor = PreprocessImages(config.dataset_dir,
                                        config.input_size[0])
        X_train, y_train, X_test, y_test = preprocessor()

    else:
        arrays = np.load(os.path.join(config.array_dir, f"arrays_{config.input_size[0]}.npz"))
        X_train = arrays['X_train']
        y_train = arrays['y_train']
        X_test = arrays['X_test']
        y_test = arrays['y_test']

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

    if not os.path.exists(config.checkpoint_dir):
        os.makedirs(config.checkpoint_dir, exist_ok=True)
    if not os.path.exists(config.log_dir):
        os.makedirs(config.log_dir, exist_ok=True)

    X_train = np.transpose(X_train, (0, 3, 1, 2))
    X_test = np.transpose(X_test, (0, 3, 1, 2))

    model = NIHXRayModel(config.model, config.input_size, config.num_classes)

    X_train = torch.Tensor(X_train)
    y_train = torch.Tensor(y_train)
    X_test = torch.Tensor(X_test)
    y_test = torch.Tensor(y_test)

    dataset = TensorDataset(X_train, y_train)
    train_num = int(len(dataset)*0.7)
    train_set, val_set = random_split(dataset, [train_num, len(dataset)-train_num])
    test_set = TensorDataset(X_test, y_test)

    scaler = GradScaler()

    func_args = (scaler, model, starttime, train_set, val_set, test_set, config)
    mp.spawn(train, args=func_args, nprocs=config.device_config.num_gpus, join=True)
