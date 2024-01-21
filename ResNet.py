# -*- coding: utf-8 -*-
"""
Created on Sun Jan 21 17:15:56 2024

@author: kevin
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as L

class ResNet9Lighting(L.LightningModule):
  ## computationnal code
    def __init__(self, in_channels, num_classes, max_lr, weight_decay):
        super().__init__()
        self.save_hyperparameters()

        self.conv1 = self.conv_block(in_channels, 64)
        self.conv2 = self.conv_block(64, 128, pool=True)
        self.res1 = nn.Sequential(self.conv_block(128, 128), self.conv_block(128, 128))

        self.conv3 = self.conv_block(128, 256, pool=True)
        self.conv4 = self.conv_block(256, 512, pool=True)
        self.res2 = nn.Sequential(self.conv_block(512, 512), self.conv_block(512, 512))

        self.classifier = nn.Sequential(nn.AdaptiveMaxPool2d(1),
                                            nn.Flatten(),
                                            nn.Dropout(0.2),
                                            nn.Linear(512, num_classes))

        self.max_lr = max_lr
        self.weight_decay = weight_decay
        
    def conv_block(self, in_channels, out_channels, pool=False):
       layers = [
           nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
           nn.BatchNorm2d(out_channels),
           nn.ReLU(inplace=True)
       ]
       if pool:
           layers.append(nn.MaxPool2d(2))
       return nn.Sequential(*layers)

    ## forward hook
    def forward(self, xb):
        out = self.conv1(xb)
        out = self.conv2(out)
        out = self.res1(out) + out
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.res2(out) + out
        out = self.classifier(out)
        return out

    ## cofigure_optimizers L hook
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.max_lr, weight_decay=self.weight_decay)
        return optimizer

    ### training step hook
    def training_step(self, batch):
      images, labels = batch
      out = self(images)                  # Generate predictions
      loss = F.cross_entropy(out, labels) # Calculate loss
      self.log("train_loss", loss, on_epoch=True)
      return loss

    ### validation_step hook #à quoi sert le validation step
    def validation_step(self, val_batch):
        images, labels = val_batch
        out = self(images)                    # Generate predictions
        loss = F.cross_entropy(out, labels)   # Calculate loss
        acc = self.accuracy(out, labels)      # Calculate accuracy
        self.log('val_loss', loss)
        return {'val_loss': loss.detach(), 'val_acc': acc}
    
    def accuracy(self, outputs, labels):
        max = torch.max(outputs)
        sum = torch.sum(max == labels)
        acc = sum.item() / len(labels)
        return torch.tensor(acc)
