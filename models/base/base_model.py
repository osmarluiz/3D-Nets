import torch
import torch.nn as nn

from models.base.initialization import initialize_weights


class BaseUNet(nn.Module):
    def __init__(self, in_channels, out_channels, inner_dims: tuple[int, ...]):
        super(BaseUNet, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.inner_dims = inner_dims

        self.encoder = self.encoder_block()
        self.decoder = self.decoder_block()

        initialize_weights(self)

    def encoder_block(self) -> nn.Module:
        raise NotImplementedError(
            "encoder_block method must be implemented by subclasses")

    def decoder_block(self) -> nn.Module:
        raise NotImplementedError(
            "decoder_block method must be implemented by subclasses")

    def forward(self, x) -> torch.Tensor:
        enc = self.encoder(x, return_tensors=True)
        output = self.decoder(enc)
        return output
