from typing import List, Union, Dict, Any

import torch
import torch.nn as nn

from models.base.initialization import initialize_weights


class BaseUNet(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, inner_dims: tuple[int, ...]):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.inner_dims = inner_dims

        self._make_encoding_layers()
        self._make_decoding_layers()
        self.pool = self.pooling_block()

        initialize_weights(self)

    def pooling_block(self) -> nn.Module:
        raise NotImplementedError(
            "pooling_block method must be implemented by subclasses")

    def conv_block(self, in_channels: int, out_channels: int) -> nn.Module:
        raise NotImplementedError(
            "conv_block method must be implemented by subclasses")

    def upconv_block(self, in_channels: int, out_channels: int) -> nn.Module:
        raise NotImplementedError(
            "upconv_block method must be implemented by subclasses")

    def _make_encoding_layers(self):
        in_channels = self.in_channels
        encoder_layers = []
        for idx, dim in enumerate(self.inner_dims[:-1]):
            module = self.conv_block(in_channels, dim)
            self.add_module(f'encoder{idx+1}', module)
            encoder_layers.append(module)
            in_channels = dim
        self.bottleneck = self.conv_block(in_channels, self.inner_dims[-1])
        self.encoder_layers = encoder_layers

    def _make_decoding_layers(self):
        decoder_steps: List[Dict[str, Any]] = []
        for idx in range(len(self.inner_dims) - 1, 0, -1):
            upconv = self.upconv_block(
                self.inner_dims[idx], self.inner_dims[idx - 1])
            decoder = self.conv_block(
                self.inner_dims[idx], self.inner_dims[idx - 1])
            self.add_module(f'upconv{idx+1}', upconv)
            self.add_module(f'decoder{idx+1}', decoder)
            decoder_steps.append(
                {'name': f'step{idx}', 'upconv': upconv, 'decoder': decoder})
        last_conv = nn.Conv3d(
            self.inner_dims[0], self.out_channels, kernel_size=1)
        self.add_module('last_conv', last_conv)
        decoder_steps.append({'name': 'last_conv', 'last_conv': last_conv})
        self.decoder_steps = decoder_steps

    def forward(self, x) -> torch.Tensor:
        encoded_tensors = []
        for idx, enc_layer in enumerate(self.encoder_layers):
            if idx == 0:
                x = enc_layer(x)
            else:
                x = enc_layer(self.pool(x))
            encoded_tensors.append((f'enc{idx+1}', x))
        x = self.bottleneck(self.pool(x))
        encoded_tensors.append((f'bottleneck', x))

        reversed_tensors = encoded_tensors[::-1]
        for idx, step in enumerate(self.decoder_steps):
            last_conv = step.get('last_conv', None)
            upconv = step.get('upconv', None)
            decoder = step.get('decoder', None)
            if last_conv:
                x = last_conv(x)
            elif upconv and decoder:
                enc_tensor = reversed_tensors[idx + 1]
                enc_x = enc_tensor[1]
                x = upconv(x)
                concat_x = torch.cat((x, enc_x), dim=1)
                x = decoder(concat_x)
            else:
                raise ValueError("Invalid decoder step")
        return x
