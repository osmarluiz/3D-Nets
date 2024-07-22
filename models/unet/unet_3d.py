import torch.nn as nn

from models.base.base_model import BaseUNet
from models.encoders.resnet import ResNetEncoder3D
from models.encoders.conv_relu import ConvReLUEncoder3D
from models.encoders.conv_bn_leakyrelu import ConvBNLeakyReLUEncoder3D


class UNet3D(BaseUNet):
    def __init__(self, in_channels: int, out_channels: int, inner_dims: tuple[int, ...],
                 encoder_type: str = 'resnet', dropout: float = 0.2):
        self.encoder_type = encoder_type
        self.dropout = dropout
        super().__init__(in_channels=in_channels,
                         out_channels=out_channels, inner_dims=inner_dims)

    def pooling_block(self) -> nn.Module:
        return nn.MaxPool3d((1, 2, 2))

    def upconv_block(self, in_channels: int, out_channels: int) -> nn.Module:
        return nn.ConvTranspose3d(in_channels, out_channels,
                                  kernel_size=(1, 2, 2), stride=(1, 2, 2))

    def conv_block(self, in_channels: int, out_channels: int) -> nn.Module:
        if self.encoder_type == 'resnet':
            return ResNetEncoder3D(in_channels, out_channels, self.dropout)
        elif self.encoder_type == 'conv_relu':
            return ConvReLUEncoder3D(in_channels, out_channels)
        elif self.encoder_type == 'conv_bn_leakyrelu':
            return ConvBNLeakyReLUEncoder3D(in_channels, out_channels, self.dropout)
        else:
            raise NotImplementedError(
                f"Encoder type {self.encoder_type} not implemented")
