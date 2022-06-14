from typing import Tuple

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.transforms import InterpolationMode

from config import INPUT_SIZE

class AugmentStage(nn.Module):
    def __init__(self, add_crop: bool = True, crop_scale: Tuple = (0.5, 0.5)):
        super(AugmentStage, self).__init__()
        layers = [
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply([transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([transforms.GaussianBlur(kernel_size=23)], p=0.5),
        ]
        if add_crop:
            layers = [transforms.RandomResizedCrop(INPUT_SIZE, scale=crop_scale, interpolation=InterpolationMode.BICUBIC)] + layers
        self.backbone = nn.Sequential(*layers)
    
    def forward(self, x):
        with torch.no_grad():
            return self.backbone(x)
