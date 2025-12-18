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
from livelossplot import PlotLosses
from tqdm import tqdm



####-------------------THE CNN ARCHITECHTURE---------------------------

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
    def __init__(self, block, layers, feat_dim=32, num_classes=16, input_channels=5):  # num_classes = out_dim for SimCLR
        super(ResNet, self).__init__()
        self.in_channels = 16

        # Initial layers
        self.conv1 = nn.Conv2d(input_channels, feat_dim//2, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(feat_dim//2)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # ResNet layers
        self.layer1 = self._make_layer(block, feat_dim//2,  layers[0])
        self.layer2 = self._make_layer(block, feat_dim, layers[1], stride=2)
        #self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        #self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        # Final pooling + fc
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(feat_dim * block.expansion, num_classes)

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

class ResNetRegression(nn.Module):
    def __init__(self, input_channels=5, emb_size=32, aux_features=4 ,out_dim=3):
        super(ResNetRegression, self).__init__()

        feat_dim = emb_size

        self.backbone = ResNet(BasicBlock, [2, 2, 2, 2], feat_dim=feat_dim, num_classes=feat_dim, input_channels=input_channels)

        
        self.head = nn.Sequential(
            nn.Linear(emb_size+aux_features, 128),
            nn.ReLU(),
            nn.Linear(128, out_dim)
        )

    def forward(self, x_img, x_aux):
        x_emb = self.backbone(x_img)

        x = torch.cat([x_emb, x_aux], dim=1)
        
        return self.head(x)
    

###---------------THE TRANSFORMER ARCHITECHTURE-------------------------



class PatchEmbedding(nn.Module):
    def __init__(self, input_channels=5, patch_size=8, emb_size=32, img_size=64):
        super().__init__()
        self.proj = nn.Conv2d(input_channels, emb_size, kernel_size=patch_size, stride=patch_size)
        self.num_patches = (img_size // patch_size) ** 2

    def forward(self, x):
        x = self.proj(x)  # [B, emb_size, n, n]
        x = x.flatten(2).transpose(1,2)  # [B, num_patches, emb_size]
        return x

class TransformerBlock(nn.Module):
    def __init__(self, emb_size=64, num_heads=4, ff_dim=128, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(emb_size)
        self.attn = nn.MultiheadAttention(emb_size, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(emb_size)
        self.ff = nn.Sequential(nn.Linear(emb_size, ff_dim), nn.GELU(), nn.Linear(ff_dim, emb_size))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        res = x
        x, _ = self.attn(self.norm1(x), self.norm1(x), self.norm1(x))
        x = res + self.dropout(x)
        res = x
        x = res + self.dropout(self.ff(self.norm2(x)))
        return x

class ViTRegression(nn.Module):
    def __init__(self, input_channels=5, emb_size=32, aux_features=4, out_dim=3, img_size=64, patch_size=8, depth=3, num_heads=4, ff_dim=128):
        super().__init__()
        self.patch_embed = PatchEmbedding(input_channels, patch_size, emb_size, img_size)
        self.cls_token = nn.Parameter(torch.zeros(1,1,emb_size))
        self.pos_embed = nn.Parameter(torch.zeros(1, 1 + (img_size//patch_size)**2, emb_size))
        self.blocks = nn.ModuleList([TransformerBlock(emb_size, num_heads, ff_dim) for _ in range(depth)])
        self.norm = nn.LayerNorm(emb_size)
        
        # head: combine CLS embedding + auxiliary features
        self.head = nn.Sequential(
            nn.Linear(emb_size + aux_features, 128),
            nn.ReLU(),
            nn.Linear(128, out_dim)
        )

    def forward(self, x_img, x_aux):
        B = x_img.shape[0]
        x = self.patch_embed(x_img)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        x = x + self.pos_embed
        for blk in self.blocks:
            x = blk(x)
        cls_emb = self.norm(x[:,0])
        # concatenate auxiliary features
        x = torch.cat([cls_emb, x_aux], dim=1)
        return self.head(x)

##-----------------SUPERVISED TRAINING MODULE-----------------------


class Supervised(object):

    def __init__(self, *args, **kwargs):
        self.args = kwargs['args']
        self.model = kwargs['model'].to(self.args.device)
        self.optimizer = kwargs['optimizer']
        self.scheduler = kwargs['scheduler']
        self.criterion = torch.nn.MSELoss().to(self.args.device)


    def train(self, train_loader, val_loader, save_model = False, folder = '', wandb_ = False, **key_name):


        liveloss = PlotLosses()

        if wandb_ == True:

            import wandb

            wandb_key = key_name.get("key")
            
            if wandb_key is None:
               raise ValueError("wandb_=True but no wandb_key provided")
        
            wandb.login(key=wandb_key)
        
            wandb.init(project=key_name("name"), config=vars(self.args))


        for epoch in tqdm(range(self.args.epochs)):

            self.model.train()
            train_loss = 0
            for imgs, feats, targs in train_loader:
                imgs, feats, targs = imgs.to(self.args.device), feats.to(self.args.device), targs.to(self.args.device)
                self.optimizer.zero_grad()
                preds = self.model(imgs, feats)
                loss = self.criterion(preds, targs)
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item() * imgs.size(0)
            train_loss /= len(train_loader.dataset)

            # Validation
            self.model.eval()
            val_loss = 0
            with torch.no_grad():
                for imgs, feats, targs in val_loader:
                    imgs, feats, targs = imgs.to(self.args.device), feats.to(self.args.device), targs.to(self.args.device)
                    preds = self.model(imgs, feats)
                    val_loss += self.criterion(preds, targs).item() * imgs.size(0)
            val_loss /= len(val_loader.dataset)

            # LiveLossPlot logging
            liveloss.update({
               'loss_train': train_loss,
               'loss_val': val_loss
            })
            liveloss.send()

            if wandb_ == True:
               wandb.log({"loss_train": loss_train, "loss_val": loss_val})
            
            if save_model ==True:
                
                if epoch%10==0:
                    save_path = folder + f"supervised_model_epoch_{epoch_counter+1}.pth"
                    torch.save(self.model.state_dict(), save_path)
                    print(f"Model saved to {save_path}")
                    #wandb.save(save_path)
         

            print(f"Epoch {epoch+1}/{self.args.epochs}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}")

        if save_model ==True:
            save_path = folder+f"supervised_model_final.pth"
            torch.save(self.model.state_dict(), save_path)
            print(f"Model saved to {save_path}")
            
        if wandb_ == True:
            #wandb.save(save_path)
            wandb.finish()
        

        

    



        
