import os
from typing import List, Dict, Tuple, Callable, Any, Optional, Union

import torch
from torch.utils.data import Dataset, DataLoader
import rasterio
import numpy as np
from PIL import Image
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from datasets.base_dataset import BaseDataset

# Define dataset class to load and preprocess the data


class CenterPivotDataset(BaseDataset):
    def __init__(self, image_dir: str, label_dir: str, extension: str = '.tiff'):
        super().__init__(image_dir, label_dir, extension)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        image_path = os.path.join(self.image_dir, self.image_files[idx])
        label_path = os.path.join(self.label_dir, self.label_files[idx])

        # Read image and label
        with rasterio.open(image_path) as src:
            image = src.read()
        label = np.array(Image.open(label_path))

        # Convert to PyTorch tensors
        image = torch.from_numpy(image).float()
        label = torch.from_numpy(label).long()

        # Correct the shape to (in_channels, 1, h, w)
        image = image.unsqueeze(1)

        return image, label
