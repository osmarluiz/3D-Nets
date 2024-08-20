import torch.nn as nn
import torch


class CNNLSTMEncoder3D(nn.Module):
    def __init__(self, input_channels: int, num_classes: int, img_size: tuple[int, int] = (512, 512),
                 depth: int = 11):
        super(CNNLSTMEncoder3D, self).__init__()
        self.img_size = img_size
        self.depth = depth

        self.cnn = nn.Sequential(
            nn.Conv3d(input_channels, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2),
            nn.Conv3d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2),
            nn.Conv3d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2)
        )

        cnn_output_size = (img_size[0] // 8) * \
            (img_size[1] // 8) * (depth // 8) * 64
        self.lstm = nn.LSTM(cnn_output_size, 128, batch_first=True)
        self.fc = nn.Linear(128, num_classes)
        self.upsample = nn.Upsample(
            size=img_size, mode='bilinear', align_corners=True)

    def forward(self, x: torch.Tensor):
        batch_size, seq_len, c, h, w = x.shape
        x = x.view(batch_size * seq_len, c, h, w)
        x = self.cnn(x)
        x = x.view(batch_size, seq_len, -1)
        x, _ = self.lstm(x)
        x = self.fc(x)
        x = x.view(batch_size * seq_len, -1, 1, 1)
        x = self.upsample(x)
        x = x.view(batch_size, seq_len, -1, h, w)
        return x
