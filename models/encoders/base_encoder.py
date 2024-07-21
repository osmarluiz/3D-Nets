from typing import List, Union

import torch
import torch.nn as nn


class BaseEncoder(nn.Module):
    def __init__(self, in_channels, inner_dims: tuple[int, ...]):
        super(BaseEncoder, self).__init__()
        self.in_channels = in_channels
        self.inner_dims = inner_dims

        self.encoder_layers = self.encoder_block()

    def encoder_block(self) -> List[nn.Module]:
        raise NotImplementedError(
            "encoder_block method must be implemented by subclasses")

    def forward(self, x, return_tensors: bool = False) -> Union[List[torch.Tensor], torch.Tensor]:
        raise NotImplementedError(
            "forward method must be implemented by subclasses")


class BaseConvEncoder(BaseEncoder):
    def __init__(self, in_channels: int, inner_dims: tuple[int, ...]):
        super(BaseConvEncoder, self).__init__(in_channels, inner_dims)
        self.pool = self.pooling_block()

    def pooling_block(self) -> nn.Module:
        raise NotImplementedError(
            "pooling_block method must be implemented by subclasses")

    def conv_block(self, in_channels: int, out_channels: int) -> nn.Module:
        raise NotImplementedError(
            "conv_block method must be implemented by subclasses")

    def encoder_block(self) -> List[nn.Module]:
        encoder_layers = []
        for dim in self.inner_dims:
            encoder_layers.append(self.conv_block(in_channels, dim))
            in_channels = dim
        return encoder_layers

    def forward(self, x, return_tensors: bool = True) -> Union[List[torch.Tensor], torch.Tensor]:
        output_tensors = []
        for idx, enc_layer in enumerate(self.encoder_layers[:-1]):
            if idx == 0:
                x = enc_layer(x)
            else:
                x = enc_layer(self.pool(x))
            output_tensors.append(x)
        output = self.encoder_layers[-1](self.pool(x))
        output_tensors.append(output)
        if return_tensors:
            return output_tensors
        else:
            return output
