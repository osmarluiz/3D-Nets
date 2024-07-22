import torch.nn as nn

from models.base.layers import ResidualBlock


class ResNetEncoder3D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int,
                 dropout: float = 0.2):
        super(ResNetEncoder3D, self).__init__()
        self.res1 = ResidualBlock(in_channels, out_channels)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.leaky_relu1 = nn.LeakyReLU(inplace=True)
        self.res2 = ResidualBlock(out_channels, out_channels)
        self.bn2 = nn.BatchNorm3d(out_channels)
        self.leaky_relu2 = nn.LeakyReLU(inplace=True)
        self.dropout = nn.Dropout3d(dropout)

    def forward(self, x):
        x = self.res1(x)
        x = self.bn1(x)
        x = self.leaky_relu1(x)
        x = self.res2(x)
        x = self.bn2(x)
        x = self.leaky_relu2(x)
        x = self.dropout(x)
        return x
