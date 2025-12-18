
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



def prepare_data():
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

    train_ = (images_train, features_scaled_train, targets_scaled_train)
    val_ = (images_val, features_scaled_val, targets_scaled_val)
    test_ = (images_test, features_scaled_test, targets_scaled_test)

    return train_, val_, test_, scaler, target_scaler



    
