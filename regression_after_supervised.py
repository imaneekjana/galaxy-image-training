import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from livelossplot import PlotLosses
from tqdm import tqdm
from supervised_module import *
from prepare_data import *

model_path = 'path/to/saved/model'

model = ResNetRegression(input_channels=5, emb_size=32, aux_features=4, out_dim=3)

#model = ViTRegression(input_channels=5, emb_size=32, aux_features=4, out_dim=3)

weights = torch.load(model_path,map_location = 'cpu')

model.load_state_dict(weights)




train_, val_, test_, scaler, target_scaler = prepare_data()

print('prepared data!')

(images_train, features_scaled_train, targets_scaled_train) = train_
(images_val, features_scaled_val, targets_scaled_val) = val_



class GalaxyDataset(Dataset):
    def __init__(self, images, features, targets):
        self.images = torch.tensor(images)           # [N, 5, 64, 64]
        self.features = torch.tensor(features)       # [N, F]
        self.targets = torch.tensor(targets)         # [N, 3]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx], self.features[idx], self.targets[idx]


dataset = GalaxyDataset(images_val, features_scaled_val, targets_scaled_val)

test_loader = DataLoader(dataset, batch_size=32, shuffle=False, drop_last = True)



device = "cuda" if torch.cuda.is_available() else "cpu"



model = model.to(device)

criterion = nn.MSELoss()

model.eval()

all_preds_specz_redshift = []
all_preds_r_sersic_index = []

all_true_specz_redshift  = []
all_true_r_sersic_index  = []

loss = 0

with torch.no_grad():
    for imgs, feats, targs in test_loader:
        imgs  = imgs.to(device)
        feats = feats.to(device)
        targs = targs.to(device)
        
        preds_scaled = model(imgs, feats)
        
        loss += criterion(preds_scaled, targs).item() * imgs.size(0)

        targs_np = target_scaler.inverse_transform(targs.cpu().numpy())
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


# Code for plotting

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

