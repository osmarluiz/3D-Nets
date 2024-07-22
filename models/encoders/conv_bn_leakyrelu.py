import torch.nn as nn


class ConvBNLeakyReLUEncoder3D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.2):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels,
                               kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.leaky_relu1 = nn.LeakyReLU(inplace=True)
        self.conv2 = nn.Conv3d(out_channels, out_channels,
                               kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm3d(out_channels)
        self.leaky_relu2 = nn.LeakyReLU(inplace=True)
        self.dropout = nn.Dropout3d(dropout)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.leaky_relu1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.leaky_relu2(x)
        x = self.dropout(x)
        return x
