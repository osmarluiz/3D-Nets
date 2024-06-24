# 3D-Nets

# Project Structure

This document explains the organization of the code for the project. The goal is to maintain a modular and scalable architecture that makes it easy to add new models, datasets, and utilities.

## Folder Structure

The project is organized as follows:

```
project/
│
├── configs/
│   ├── config.yaml
│
├── datasets/
│   ├── __init__.py
│   ├── base_dataset.py
│   └── center_pivot_dataset.py
│
├── models/
│   ├── __init__.py
│   ├── base/
│   │   ├── __init__.py
│   │   ├── base_model.py
│   │   ├── layers.py
│   │   └── initialization.py
│   │
│   ├── encoders/
│   │   ├── __init__.py
│   │   ├── resnet.py
│   │   └── efficientnet.py
│   │
│   ├── unet/
│   │   ├── __init__.py
│   │   ├── unet_2d.py
│   │   └── unet_3d.py
│   │
│   └── utils/
│       ├── __init__.py
│       ├── model_utils.py
│       ├── logger.py
│       └── callbacks.py
│
├── train.py
├── evaluate.py
└── trainer.py
```

### `configs/`

- **config.yaml**: This file contains configuration parameters for the project, such as paths to datasets, model parameters, and training parameters.

### `datasets/`

- **`__init__.py`**: Initializes the datasets module.
- **base_dataset.py**: Contains the base dataset class which other dataset classes inherit from.
- **center_pivot_dataset.py**: Contains dataset classes specifically for the center pivot dataset.

### `models/`

- **`__init__.py`**: Initializes the models module.
- **base/**: Contains base classes and layers that are shared across different models.
  - **`__init__.py`**: Initializes the base submodule.
  - **base_model.py**: Contains the base model class.
  - **layers.py**: Contains custom layers, such as residual blocks.
  - **initialization.py**: Contains functions for initializing model weights.
- **encoders/**: Contains different encoder architectures that can be used with the models.
  - **`__init__.py`**: Initializes the encoders submodule.
  - **resnet.py**: Contains the ResNet encoder implementation.
  - **efficientnet.py**: Contains the EfficientNet encoder implementation.
- **unet/**: Contains U-Net model implementations.
  - **`__init__.py`**: Initializes the U-Net submodule.
  - **unet_2d.py**: Contains the 2D U-Net model implementation.
  - **unet_3d.py**: Contains the 3D U-Net model implementation.
- **utils/**: Contains utility functions and classes for the models.
  - **`__init__.py`**: Initializes the utils submodule.
  - **model_utils.py**: Contains utility functions for model evaluation and metric computation.
  - **logger.py**: Contains logging utility functions.
  - **callbacks.py**: Contains callback classes, such as early stopping.

### `train.py`

This script contains the logic for training the model. It initializes the datasets, dataloaders, model, and training loop.

### `evaluate.py`

This script contains the logic for evaluating the model on the test dataset. It loads the trained model and computes metrics.

### `trainer.py`

This script contains the `Trainer` class, which encapsulates the training and evaluation logic. This class makes it easier to manage the training process and keeps the training script clean.

## How to Use

### Training

To train the model, simply run:

```bash
python train.py
```

### Evaluation

To evaluate the model, run:

```bash
python evaluate.py
```

### Configurations

You can modify the `configs/config.yaml` file to change the dataset paths, model parameters, and training parameters.

```yaml
train:
  image_dir: 'path/to/train/images'
  mask_dir: 'path/to/train/masks'
  batch_size: 4
  num_workers: 4
  num_epochs: 20
  learning_rate: 1e-4
  use_residual: true

val:
  image_dir: 'path/to/val/images'
  mask_dir: 'path/to/val/masks'

test:
  image_dir: 'path/to/test/images'
  mask_dir: 'path/to/test/masks'

model:
  type: 'unet_2d'
  in_channels: 11
  out_channels: 1
```

## Adding New Models

To add a new model, follow these steps:

1. **Create the Model File**: Add a new file in the `models/` directory (or appropriate subdirectory) for your model.
2. **Implement the Model**: Implement your model class in the new file.
3. **Import the Model**: Import your model in `models/__init__.py` to make it available for use.

## Adding New Datasets

To add a new dataset, follow these steps:

1. **Create the Dataset File**: Add a new file in the `datasets/` directory for your dataset.
2. **Implement the Dataset**: Implement your dataset class in the new file, inheriting from `BaseDataset`.
3. **Import the Dataset**: Import your dataset in `datasets/__init__.py` to make it available for use.

## Contribution Guidelines

- Follow the existing code style and organization.
- Write docstrings for all functions and classes.
- Test your code before pushing changes.
