import torch
import torch.nn as nn
from torchvision.models import resnet34

class CustomCNN(nn.Module):
    def __init__(self, encoder_features: int = 6400):
        super(CustomCNN, self).__init__()
        # Nuñez-Alcover, A., León, P. J., & Calvo-Zaragoza, J.
        # Glyph and position classification of music symbols in early music manuscripts
        layers = [
            nn.Conv2d(3, 32, 3, padding="same"),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding="same"),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            nn.Dropout(0.25),
            nn.Conv2d(32, 64, 3, padding="same"),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding="same"),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            nn.Dropout(0.25),
            nn.Flatten()
        ]
        if encoder_features != 6400:
            layers.append(nn.Linear(6400, encoder_features))
        self.backbone = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.backbone(x)

class ResnetEncoder(nn.Module):
    def __init__(self):
        super(ResnetEncoder, self).__init__()
        resnet = resnet34(pretrained=True)
        resnet_modules = list(resnet.children())
        self.backbone = nn.Sequential(*resnet_modules[:-1], nn.Flatten())

    def forward(self, x):
        return self.backbone(x)

class ExpanderLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super(ExpanderLayer, self).__init__()
        self.linear_layer = nn.Linear(in_features, out_features)
        self.bn = nn.BatchNorm1d(out_features)
        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.linear_layer(x)
        x = self.bn(x)
        x = self.activation(x)
        return x

class VICReg(nn.Module):
    def __init__(self, encoder_features: int = 1600, expander_features: int = 1024):
        super(VICReg, self).__init__()
        self.encoder = CustomCNN(encoder_features=encoder_features)
        self.expander = nn.Sequential(
            ExpanderLayer(encoder_features, expander_features),
            ExpanderLayer(expander_features, expander_features),
            nn.Linear(expander_features, expander_features)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.expander(x)
        return x

    def save(self, path):
        torch.save(self.encoder.state_dict(), path)
