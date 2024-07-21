import torch
import torch.nn as nn

from models.base.base_model import BaseUNet
from models.encoders.base_encoder import BaseConvEncoder
from models.encoders.resnet import ResNetEncoder3D
from models.encoders.conv_relu import ConvReLUEncoder3D
from models.encoders.conv_bn_leakyrelu import ConvBNLeakyReLUEncoder3D


class UNet3D(BaseUNet):
    def __init__(self, in_channels: int, out_channels: int, inner_dims: tuple[int, ...],
                 encoder_type: str = 'resnet', dropout: float = 0.2):
        super(UNet3D, self).__init__(in_channels, out_channels, inner_dims)
        self.encoder_type = encoder_type
        self.dropout = dropout

    def conv_block(self, in_channels: int, out_channels: int) -> nn.Module:
        return self.encoder_block().conv_block(in_channels, out_channels)

    def encoder_block(self) -> BaseConvEncoder:
        if self.encoder_type == 'resnet':
            return ResNetEncoder3D(self.in_channels, self.inner_dims, self.dropout)
        elif self.encoder_type == 'conv_relu':
            return ConvReLUEncoder3D(self.in_channels, self.inner_dims)
        elif self.encoder_type == 'conv_bn_leakyrelu':
            return ConvBNLeakyReLUEncoder3D(self.in_channels, self.inner_dims)
        else:
            raise NotImplementedError(
                f"Encoder type {self.encoder_type} not implemented")

    def decoder_block(self) -> nn.Module:
        decoder_layers = []
        for idx in range(len(self.inner_dims) - 1, 0, -1):
            decoder_layers.append(
                nn.ConvTranspose3d(self.inner_dims[idx], self.inner_dims[idx - 1],
                                   kernel_size=(1, 2, 2), stride=(1, 2, 2)))
            decoder_layers.append(
                self.conv_block(self.inner_dims[idx], self.inner_dims[idx - 1]))
        decoder_layers.append(
            nn.Conv3d(self.inner_dims[-1], self.out_channels, kernel_size=1))
        return nn.Sequential(*decoder_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        encoder_output = self.encoder(x, return_tensors=True)
        decoder_output = self.decoder(encoder_output)
        return decoder_output
