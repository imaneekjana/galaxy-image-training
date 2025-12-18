## Necessary Imports

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



## The basic block architechture

class BasicBlock(nn.Module):
    expansion = 1  # keeps output channels same as input

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += identity
        out = self.relu(out)

        return out
    

    
## ResNet backbone architecture

class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=32, input_channels=5):  # num_classes = out_dim for SimCLR
        super(ResNet, self).__init__()
        self.in_channels = 16

        # Initial layers
        self.conv1 = nn.Conv2d(input_channels, 16, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # ResNet layers
        self.layer1 = self._make_layer(block, 16,  layers[0])
        self.layer2 = self._make_layer(block, 32, layers[1], stride=2)
        #self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        #self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        # Final pooling + fc
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(32 * block.expansion, num_classes)

    def _make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * block.expansion),
            )

        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        #x = self.layer3(x)
        #x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x



    
class ResNetSimCLR(nn.Module):
    def __init__(self, base_model="resnet18", out_dim=16):
        super(ResNetSimCLR, self).__init__()

        if base_model == "resnet18":
            self.backbone = ResNet(BasicBlock, [2, 2, 2, 2], num_classes=out_dim)
        elif base_model == "resnet50":
            # You’d need to define Bottleneck block like torchvision
            raise NotImplementedError("ResNet50 requires Bottleneck block")
        else:
            raise ValueError("Invalid base model")

        dim_mlp = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Linear(dim_mlp, dim_mlp),
            nn.ReLU(),
            nn.Linear(dim_mlp, out_dim)
        )

    def forward(self, x):
        return self.backbone(x)

    




# The SimCLR training protocol


class SimCLR(object):

    def __init__(self, *args, **kwargs):
        self.args = kwargs['args']
        self.model = kwargs['model'].to(self.args.device)
        self.optimizer = kwargs['optimizer']
        self.scheduler = kwargs['scheduler']
        self.criterion = torch.nn.CrossEntropyLoss().to(self.args.device)

    def info_nce_loss(self, features):

        labels = torch.cat([torch.arange(self.args.batch_size) for i in range(self.args.n_views)], dim=0)
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        labels = labels.to(self.args.device)

        features = F.normalize(features, dim=1)

        similarity_matrix = torch.matmul(features, features.T)
        # assert similarity_matrix.shape == (
        #     self.args.n_views * self.args.batch_size, self.args.n_views * self.args.batch_size)
        # assert similarity_matrix.shape == labels.shape

        # discard the main diagonal from both: labels and similarities matrix
        mask = torch.eye(labels.shape[0], dtype=torch.bool).to(self.args.device)
        labels = labels[~mask].view(labels.shape[0], -1)
        similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)
        # assert similarity_matrix.shape == labels.shape

        # select and combine multiple positives
        positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)

        # select only the negatives the negatives
        negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

        logits = torch.cat([positives, negatives], dim=1)
        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(self.args.device)

        logits = logits / self.args.temperature
        return logits, labels
    
    def test(self, loader,desc=''):
        
        modelh = self.model
        
        modelh.eval()
        
        loss_ = 0
        
        with torch.no_grad():
        
            for image1, image2,_,_ in loader:
                images = (image1,image2)
                
                images = torch.cat(images, dim=0)

                images = images.to(self.args.device)

                with autocast(enabled=self.args.fp16_precision):
                    features = modelh(images)
                    logits, labels = self.info_nce_loss(features)
                    loss = self.criterion(logits, labels)
                    loss_ += loss
        
        return loss_/len(loader)
    
    
    
            
        
        

    def train(self, train_loader, val_loader, save_model = False, folder = '', wandb_ = False, **key_name):

        scaler = GradScaler(enabled=self.args.fp16_precision)

        liveloss = PlotLosses()

        if wandb_ == True:

            import wandb

            wandb_key = key_name.get("key")
            
            if wandb_key is None:
               raise ValueError("wandb_=True but no wandb_key provided")
        
            wandb.login(key=wandb_key)
        
            wandb.init(project=key_name("name"), config=vars(self.args))

        
        n_iter = 0
    

        for epoch_counter in tqdm(range(self.args.epochs),desc='epoch'):
            loss_train = 0
            for image1, image2,_,_ in train_loader:
                images = (image1,image2)
                
                images = torch.cat(images, dim=0)

                images = images.to(self.args.device)

                with autocast(enabled=self.args.fp16_precision):
                    features = self.model(images)
                    logits, labels = self.info_nce_loss(features)
                    loss = self.criterion(logits, labels)
                    loss_train += loss.item()

                self.optimizer.zero_grad()

                scaler.scale(loss).backward()

                scaler.step(self.optimizer)
                scaler.update()
                
                n_iter += 1
            
            #loss_train = self.test(train_loader).item()
            loss_train = loss_train/len(train_loader)
            loss_val = self.test(val_loader,desc='validation').item()
            
            # LiveLossPlot logging
            liveloss.update({
                'loss_train': loss_train,
                'loss_val': loss_val
             })
            liveloss.send()

            if wandb_ == True:
                wandb.log({"loss_train": loss_train, "loss_val": loss_val})
            

            if save_model ==True:
                
                if epoch_counter%10==0:
                    save_path = folder + f"simclr_model_epoch_{epoch_counter+1}.pth"
                    torch.save(self.model.state_dict(), save_path)
                    print(f"Model saved to {save_path}")
                    #wandb.save(save_path)
               

        save_path = folder+f"simclr_model_final.pth"
        torch.save(self.model.state_dict(), save_path)
        print(f"Model saved to {save_path}")

        if wandb_ == True:
            wandb.save(save_path)
            wandb.finish()

        
        
        
            
       