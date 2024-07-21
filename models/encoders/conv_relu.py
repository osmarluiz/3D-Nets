import torch.nn as nn

from models.encoders.base_encoder import BaseConvEncoder


class ConvReLUEncoder3D(BaseConvEncoder):
    def __init__(self, in_channels: int, inner_dims: tuple[int, ...]):
        super(ConvReLUEncoder3D, self).__init__(in_channels, inner_dims)

    def pooling_block(self) -> nn.Module:
        return nn.MaxPool3d((1, 2, 2))

    def conv_block(self, in_channels: int, out_channels: int) -> nn.Module:
        return nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
