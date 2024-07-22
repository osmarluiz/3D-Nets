import os
from typing import Tuple

import torch
from torch.utils.data import Dataset
import numpy as np
from PIL import Image


# Define dataset class to load and preprocess the data
class BaseDataset(Dataset):
    def __init__(self, image_dir: str, label_dir: str, extension: str = '.tiff'):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.image_files = [f for f in os.listdir(
            image_dir) if f.endswith(extension)]
        self.label_files = [f for f in os.listdir(
            label_dir) if f.endswith(extension)]

    def __len__(self) -> int:
        return len(self.image_files)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        image_path = os.path.join(self.image_dir, self.image_files[idx])
        label_path = os.path.join(self.label_dir, self.label_files[idx])

        # Read image and label
        image = np.array(Image.open(image_path))
        label = np.array(Image.open(label_path))

        # Convert to PyTorch tensors
        image = torch.from_numpy(image).float()
        label = torch.from_numpy(label).long()

        # Correct the shape to (in_channels, 1, h, w)
        image = image.unsqueeze(1)

        return image, label
