import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
import timm

class SpatialAttention(nn.Module):
     def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

     def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)   # average along channel
        max_out, _ = torch.max(x, dim=1, keepdim=True) # max along channel
        concat = torch.cat([avg_out, max_out], dim=1)
        attention = self.sigmoid(self.conv(concat))
        return x * attention
class InterpretationFusionNet(nn.Module):
    """
    Takes original image (3 channels) and two interpretation maps (2 channels),
    outputs two expert weights.
    """
    def __init__(self, in_channels=5,out_dim=2,cnn_channels=[32, 64, 128],spatial_attn_kernel=7):
         super().__init__()
         layers=[]
         prev = in_channels

         for ch in cnn_channels:
             layers.append(nn.Conv2d(prev,ch,kernel_size=3,stride=2,padding=1))
             layers.append(nn.BatchNorm2d(ch))
             layers.append(nn.ReLU(inplace=True))
             prev = ch
         self.cnn = nn.Sequential(*layers)
         self.spatial_attn=SpatialAttention(kernel_size=spatial_attn_kernel)
         self.global_pool=nn.AdaptiveAvgPool2d(1)

         self.mlp = nn.Sequential(
            nn.Linear(cnn_channels[-1], 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, out_dim)
        )
    def forward(self, x, cam1, cam2):
        """
        x: (B, 3, H, W) original image
        cam1: (B, 1, H, W) heatmap from expert 1
        cam2: (B, 1, H, W) heatmap from expert 2
        """
        # Concatenate along channel dimension
        combined = torch.cat([x, cam1, cam2], dim=1)   # (B, 5, H, W)
        features = self.cnn(combined)                  # (B, C, H', W')
        attended = self.spatial_attn(features)         # (B, C, H', W')
        pooled = self.global_pool(attended)            # (B, C, 1, 1)
        vec = pooled.view(pooled.size(0), -1)          # (B, C)
        logits = self.mlp(vec)                         # (B, 2)
        weights = F.softmax(logits, dim=1)             # (B, 2)
        return weights

