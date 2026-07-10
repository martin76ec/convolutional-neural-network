import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.25):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout2d(dropout)

    def forward(self, x):
        x = F.relu(self.bn(self.conv(x)))
        x = self.pool(x)
        x = self.dropout(x)
        return x


class CNN(nn.Module):
    def __init__(self, num_classes: int, base_channels: int = 32, dropout: float = 0.25):
        super().__init__()

        self.block1 = ConvBlock(3, base_channels, dropout)
        self.block2 = ConvBlock(base_channels, base_channels * 2, dropout)
        self.block3 = ConvBlock(base_channels * 2, base_channels * 4, dropout)
        self.block4 = ConvBlock(base_channels * 4, base_channels * 8, dropout)

        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(base_channels * 8, 256)
        self.fc2 = nn.Linear(256, num_classes)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)

        x = self.gap(x).squeeze(-1).squeeze(-1)
        x = F.relu(self.fc1(x))
        x = self.drop(x)
        x = self.fc2(x)
        return x