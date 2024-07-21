import torch.nn as nn

from models.base.layers import ResidualBlock
from models.encoders.base_encoder import BaseConvEncoder


class ResNetEncoder3D(BaseConvEncoder):
    def __init__(self, in_channels: int, inner_dims: tuple[int, ...],
                 dropout: float = 0.2):
        super(ResNetEncoder3D, self).__init__(in_channels, inner_dims)
        self.dropout = dropout

    def pooling_block(self) -> nn.Module:
        return nn.MaxPool3d((1, 2, 2))

    def conv_block(self, in_channels: int, out_channels: int) -> nn.Module:
        return nn.Sequential(
            ResidualBlock(in_channels, out_channels),
            nn.BatchNorm3d(out_channels),
            nn.LeakyReLU(inplace=True),
            ResidualBlock(out_channels, out_channels),
            nn.BatchNorm3d(out_channels),
            nn.LeakyReLU(inplace=True),
            nn.Dropout3d(self.dropout)
        )
