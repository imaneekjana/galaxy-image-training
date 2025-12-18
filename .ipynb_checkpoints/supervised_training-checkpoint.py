
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
from supervised_module import *
from prepare_data import *

# Prepare the data
train_, val_, test_, scaler, target_scaler = prepare_data()

print('prepared data!')

(images_train, features_scaled_train, targets_scaled_train) = train_
 
#val_ = (images_val, features_scaled_val, targets_scaled_val)
#test_ = (images_test, features_scaled_test, targets_scaled_test)



class GalaxyDataset(Dataset):
    def __init__(self, images, features, targets):
        self.images = torch.tensor(images)           # [N, 5, 64, 64]
        self.features = torch.tensor(features)       # [N, F]
        self.targets = torch.tensor(targets)         # [N, 3]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx], self.features[idx], self.targets[idx]



train_dataset = GalaxyDataset(images_train, features_scaled_train, targets_scaled_train)

##dataset constuction

dataset = train_dataset

train_size = int(0.80 * len(dataset))
val_size = int(0.1 * len(dataset))
test_size = len(dataset) - val_size - train_size

train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, drop_last = True)
val_loader = DataLoader(val_dataset, batch_size=32, drop_last = True)
test_loader = DataLoader(test_dataset, batch_size=32, drop_last = True)

print('prepared datasets!')

## Train the Supervised model!!


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

parser.add_argument('--epochs', default=2, type=int, metavar='N',
                    help='number of total epochs to run')


parser.add_argument('-b', '--batch-size', default=32, type=int,
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

parser.add_argument('--out_dim', default=3, type=int,
                    help='number of variables to be regressed')

parser.add_argument('--emb_size', default=32, type=int,
                    help='feature dimension of internal embedding space')

parser.add_argument('--log-every-n-steps', default=100, type=int,
                    help='Log every n steps')

parser.add_argument('--temperature', default=0.07, type=float,
                    help='softmax temperature (default: 0.07)')

parser.add_argument('--n-views', default=2, type=int, metavar='N',
                    help='Number of views for contrastive learning training.')

parser.add_argument('--gpu-index', default=0, type=int, help='Gpu index.')


args, unknown = parser.parse_known_args()


# check if gpu training is available

if not args.disable_cuda and torch.cuda.is_available():
    args.device = torch.device('cuda')
    torch.backends.cudnn.deterministic = True
    cudnn.benchmark = True    
else:
    args.device = torch.device('cpu')
    args.gpu_index = -1


#model = ResNetRegression(input_channels=5, emb_size=args.emb_size, aux_features=4, out_dim=args.out_dim)
model = ViTRegression(input_channels=5, emb_size=args.emb_size, aux_features=4, out_dim=args.out_dim)


# Move to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

optimizer = torch.optim.Adam(model.parameters(), args.lr, weight_decay=args.weight_decay)

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(train_loader), eta_min=0, last_epoch=-1)

    #  It’s a no-op if the 'gpu_index' argument is a negative integer or None.


with torch.cuda.device(args.gpu_index):
    
    supervised = Supervised(model=model, optimizer=optimizer, scheduler=scheduler, args=args)

    
    
    supervised.train(train_loader, val_loader, save_model=False, folder = '', wandb_=False)
    

    print("done")
    









