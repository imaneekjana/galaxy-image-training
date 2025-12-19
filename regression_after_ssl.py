## Imports

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
from prepare_data import *


# Give path to the saved ResNet encoder model

encoder_path = 'path/to/saved/model'

weights = torch.load(encoder_path, map_location='cpu')

# Create the architecture

encoder = ResNetSimCLR(input_channels=5, feat_dim=32, out_dim=10)

encoder.load_state_dict(weights)

#removing the projection head

backbone = encoder.backbone
backbone.fc = nn.Identity()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
encoder = encoder.to(device)

# Freeze the encoder weights 

for p in encoder.parameters():
    p.requires_grad = False

encoder.eval()


# Add a regressor

latent_dim = 32
hidden_dim = 128
num_classes = 3

regressor = nn.Sequential(
    nn.Linear(latent_dim+4, hidden_dim),
    nn.ReLU(),
    nn.Linear(hidden_dim,hidden_dim),
    nn.ReLU(),
    nn.Linear(hidden_dim,num_classes) 
)

# Prepare data

train_, val_, test_, scaler, target_scaler = prepare_data()

print('prepared data!')

(images_train, features_scaled_train, targets_scaled_train) = train_
(images_val, features_scaled_val, targets_scaled_val) = val_

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
            return xi, xj, img, (feature, target)
        else:
            return img, feature, target




# Data for training
train_dataset = GalaxyDataset(images_train, features_scaled_train, targets_scaled_train)
dataset = train_dataset

train_size = int(0.6 * len(dataset))
val_size = int(0.2 * len(dataset))
test_size = len(dataset) - val_size - train_size

train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

# chose batchsize = 128 while simclr training and choose batchsize = 32 while regressor training
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, drop_last = True)
val_loader = DataLoader(val_dataset, batch_size=32, drop_last = True)
#test_loader = DataLoader(test_dataset, batch_size=32, drop_last = True)




# Data for testing
dataset = GalaxyDataset(images_val, features_scaled_val, targets_scaled_val)
test_loader = DataLoader(dataset, batch_size=32, shuffle=False, drop_last = True)


# Train the regressor (optionally load it)

regressor = train_regressor(regressor, encoder, train_loader, val_loader, epochs=50, save_model = False, file_path = '')

'''
weights_reg = torch.load('path/to/regressor', map_location = 'cpu')

regressor.load_state_dict(weights_reg)

regressor = regressor.to(device)

criterion = nn.MSELoss()  # multi-task regression

'''



with torch.no_grad():

    loss = 0

    all_preds_specz_redshift = []
    all_preds_r_sersic_index = []

    all_true_specz_redshift  = []
    all_true_r_sersic_index  = []

    

    for images, features, targets in test_loader:

        images = images.to(device)
        features_aux = features.to(device)
        targets = targets.to(device)
        
        with torch.no_grad():
            feats = encoder(images)
            
        x_embed = torch.cat([features_aux, feats], dim=1)

        preds_scaled = regressor(x_embed).squeeze()

        loss_ = criterion(preds_scaled, targets)

        loss += loss_.item() * images.size(0)

        targs_np = target_scaler.inverse_transform(targets.cpu().numpy())
        targs = torch.tensor(targs_np, dtype=torch.float32)

        preds_np = target_scaler.inverse_transform(preds_scaled.cpu().numpy())
        preds = torch.tensor(preds_np, dtype=torch.float32)

        all_preds_specz_redshift.append(preds[:, 0].cpu())   # specz_redshift prediction
        all_true_specz_redshift.append(targs[:, 0].cpu())    # specz_redshift ground truth

        all_preds_r_sersic_index.append(preds[:, 1].cpu())   # specz_redshift prediction
        all_true_r_sersic_index.append(targs[:, 1].cpu())    # specz_redshift ground truth

print('loss : ', loss/len(test_loader.dataset))

# concatenate to 1D tensors
import torch

all_preds_specz_redshift = torch.cat(all_preds_specz_redshift)
all_preds_r_sersic_index = torch.cat(all_preds_r_sersic_index)

all_true_specz_redshift  = torch.cat(all_true_specz_redshift)
all_true_r_sersic_index  = torch.cat(all_true_r_sersic_index)


import matplotlib.pyplot as plt

plt.figure(figsize=(6,6))

plt.scatter(all_true_specz_redshift, all_preds_specz_redshift, s=5, alpha=0.5)
plt.plot([all_true_specz_redshift.min(), all_true_specz_redshift.max()],[all_true_specz_redshift.min(), all_true_specz_redshift.max()],linewidth=2)

plt.xlabel("True Redshift", fontsize = 20)
plt.ylabel("Predicted Redshift", fontsize = 20)
#plt.title("Predicted vs True Redshift")

plt.grid(True)
plt.show()

plt.figure(figsize=(6,6))

plt.scatter(all_true_r_sersic_index, all_preds_r_sersic_index, s=5, alpha=0.5)
plt.plot([all_true_r_sersic_index.min(), all_true_r_sersic_index.max()],[all_true_r_sersic_index.min(), all_true_r_sersic_index.max()],linewidth=2)

plt.xlabel("True r_sersic_index", fontsize=20)
plt.ylabel("Predicted r_sersic_index", fontsize=20)
#plt.title("Predicted vs True r_sersic_index", fontsize=25)

plt.grid(True)
plt.show()


        

        











