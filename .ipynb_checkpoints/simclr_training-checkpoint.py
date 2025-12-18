
import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from torch.cuda.amp import GradScaler, autocast
import logging
import os
import sys
from livelossplot import PlotLosses
from tqdm import tqdm
from simclr_module import *





## Load the data and normalize it

f_train = h5py.File("5x64x64_training_with_morphology.hdf5", "r")
f_val = h5py.File("5x64x64_validation_with_morphology.hdf5", "r")
f_test = h5py.File("5x64x64_testing_with_morphology.hdf5", "r")



# Images (5 bands)
images_train = f_train['image'][:]  # [N, 5, 64, 64]
images_val = f_val['image'][:]  # [N, 5, 64, 64]
images_test = f_test['image'][:]  # [N, 5, 64, 64]

# Optional photometric features to use as auxiliary inputs
features_train = np.stack([
    f_train['r_cmodel_mag'][:],
    f_train['r_ellipticity'][:],
    f_train['specz_mag_i'][:],
    f_train['r_peak_surface_brightness'][:]
], axis=1)  # shape [N, 4]

features_test = np.stack([
    f_test['r_cmodel_mag'][:],
    f_test['r_ellipticity'][:],
    f_test['specz_mag_i'][:],
    f_test['r_peak_surface_brightness'][:]
], axis=1)  # shape [N, 4]

features_val = np.stack([
    f_val['r_cmodel_mag'][:],
    f_val['r_ellipticity'][:],
    f_val['specz_mag_i'][:],
    f_val['r_peak_surface_brightness'][:]
], axis=1)  # shape [N, 4]


# Targets: redshift + r-band morphology (multi-task regression)
targets_train = np.stack([
    f_train['specz_redshift'][:],
    f_train['r_sersic_index'][:],
    f_train['r_half_light_radius'][:]
], axis=1)  # shape [N, 3]

targets_val = np.stack([
    f_val['specz_redshift'][:],
    f_val['r_sersic_index'][:],
    f_val['r_half_light_radius'][:]
], axis=1)  # shape [N, 3]

targets_test = np.stack([
    f_test['specz_redshift'][:],
    f_test['r_sersic_index'][:],
    f_test['r_half_light_radius'][:]
], axis=1)  # shape [N, 3]

# 3️⃣ Data preprocessing

# Normalize images (0-1)
images_test = images_test.astype(np.float32)
#images_test = images_test / images_train.max()

images_val = images_val.astype(np.float32)
#images_val = images_val / images_tr.max()

images_train = images_train.astype(np.float32)
#images_train = images_train / images_train.max()

# Scale auxiliary features
scaler = StandardScaler()
features_scaled_train = scaler.fit_transform(features_train).astype(np.float32)
features_scaled_val   = scaler.transform(features_val).astype(np.float32)
features_scaled_test  = scaler.transform(features_test).astype(np.float32)


# Targets scaling (optional)
target_scaler = StandardScaler()
targets_scaled_train = target_scaler.fit_transform(targets_train).astype(np.float32)
targets_scaled_val = target_scaler.transform(targets_val).astype(np.float32)
targets_scaled_test = target_scaler.transform(targets_test).astype(np.float32)


## Data augmentation module

import torchvision.transforms as T
import torchvision.transforms.functional as F
import random

class AddGaussianNoise(object):
    def __init__(self, mean=0., std=0.05):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        noise = torch.randn_like(tensor) * self.std + self.mean
        return tensor + noise


class GalaxySimCLRAugment:
    """
    SimCLR-style augmentations for galaxy images.
    Includes: random crop, resize, flips, Gaussian noise.
    """

    def __init__(self, crop_size=64):
        self.crop_size = crop_size

        self.base_transforms = T.Compose([
            # Random crop (with padding to avoid degeneracy)
            T.RandomResizedCrop(
                size=crop_size,
                scale=(0.7, 1.0),          # moderate crop to preserve structure
                ratio=(0.9, 1.1)
            ),

            # Flips
            T.RandomHorizontalFlip(p=0.5),
            T.RandomVerticalFlip(p=0.5),

            # Convert to tensor if needed
            T.ConvertImageDtype(torch.float32),

            # Gaussian noise
            AddGaussianNoise(std=0.03),
        ])

    def __call__(self, img):
        # produce two views independently
        xi = self.base_transforms(img)
        xj = self.base_transforms(img)
        return xi, xj

## Dataset preparation

## Preparing the dataset --- 

class GalaxyDataset(torch.utils.data.Dataset):
    def __init__(self, images, features, targets, augment=None):
        self.images = images
        self.features = features
        self.targets = targets
        self.augment = augment

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx]        # tensor [5, H, W] or numpy
        feature = self.features[idx]
        target = self.targets[idx]

        if isinstance(img, np.ndarray):
            img = torch.tensor(img, dtype=torch.float32)
            feature = torch.tensor(feature, dtype=torch.float32)
            target = torch.tensor(target, dtype=torch.float32)

        if self.augment:
            xi, xj = self.augment(img)
            return xi, xj, img, feature, target
        else:
            return img, feature, target


augment = GalaxySimCLRAugment(crop_size=64)


## Customize loaders according to your need

train_dataset = GalaxyDataset(images_train, features_scaled_train, targets_scaled_train, augment=augment)
dataset = train_dataset
train_size = int(0.6 * len(dataset))
val_size = int(0.2 * len(dataset))
test_size = len(dataset) - val_size - train_size
train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

## chose batchsize = 128 while simclr training and choose batchsize = 32 while regressor training

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, drop_last = True)
val_loader = DataLoader(val_dataset, batch_size=32, drop_last = True)
test_loader = DataLoader(test_dataset, batch_size=32, drop_last = True)


## Train the SimCLR model!!


from torch.utils.tensorboard import SummaryWriter
import argparse
import torch
import torch.backends.cudnn as cudnn
from torchvision import models

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch SimCLR')

#parser.add_argument('-data', metavar='DIR', default='./datasets',help='path to dataset')
'''
parser.add_argument('-dataset-name', default='stl10',
                    help='dataset name', choices=['stl10', 'cifar10'])

parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18',
                    choices=model_names,
                    help='model architecture: ' +
                         ' | '.join(model_names) +
                         ' (default: resnet50)')'''

parser.add_argument('-j', '--workers', default=12, type=int, metavar='N',
                    help='number of data loading workers (default: 32)')

parser.add_argument('--epochs', default=60, type=int, metavar='N',
                    help='number of total epochs to run')


parser.add_argument('-b', '--batch-size', default=128, type=int,
                    metavar='N',
                    help='mini-batch size, this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')


parser.add_argument('--lr', '--learning-rate', default=0.0003, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')

parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')

parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')

parser.add_argument('--disable-cuda', action='store_true',
                    help='Disable CUDA')

parser.add_argument('--fp16-precision', action='store_true',
                    help='Whether or not to use 16-bit precision GPU training.')

parser.add_argument('--out_dim', default=16, type=int,
                    help='latent space dimension where constrastive loss is applied')

parser.add_argument('--feat_dim', default=32, type=int,
                    help='feature dimension')

parser.add_argument('--log-every-n-steps', default=100, type=int,
                    help='Log every n steps')

parser.add_argument('--temperature', default=0.07, type=float,
                    help='softmax temperature (default: 0.07)')

parser.add_argument('--n-views', default=2, type=int, metavar='N',
                    help='Number of views for contrastive learning training.')

parser.add_argument('--gpu-index', default=0, type=int, help='Gpu index.')


args, unknown = parser.parse_known_args()

assert args.n_views == 2, "Only two view training is supported. Please use --n-views 2."
    # check if gpu training is available
if not args.disable_cuda and torch.cuda.is_available():
    args.device = torch.device('cuda')
    torch.backends.cudnn.deterministic = True
    cudnn.benchmark = True    
else:
    args.device = torch.device('cpu')
    args.gpu_index = -1


model = ResNetSimCLR(input_channels=5, feat_dim=args.feat_dim, out_dim=args.out_dim)

# Move to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

optimizer = torch.optim.Adam(model.parameters(), args.lr, weight_decay=args.weight_decay)

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(train_loader), eta_min=0,
                                                           last_epoch=-1)

    #  It’s a no-op if the 'gpu_index' argument is a negative integer or None.


with torch.cuda.device(args.gpu_index):
    
    simclr = SimCLR(model=model, optimizer=optimizer, scheduler=scheduler, args=args)
    
    simclr.train(train_loader, val_loader, save_model=True, folder = '', wandb_=False)
    










