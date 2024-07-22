import yaml
import torch
from torch.utils.data import DataLoader
from models.unet.unet_3d import UNet3D
from trainer import Trainer
from datasets.center_pivot_dataset import CenterPivotDataset


def load_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config


def main(config_path):
    config = load_config(config_path)

    train_image_dir = config['train']['image_dir']
    train_label_dir = config['train']['mask_dir']
    val_image_dir = config['val']['image_dir']
    val_label_dir = config['val']['mask_dir']
    batch_size = config['train']['batch_size']
    num_workers = config['train']['num_workers']

    in_channels = config['model']['in_channels']
    out_channels = config['model']['out_channels']
    encoder_type = config['model']['encoder_type']
    inner_dims = config['model']['inner_dims']

    lr = config['train']['learning_rate']
    num_epochs = config['train']['num_epochs']

    best_model_path = config['paths']['best_model_path']

    train_dataset = CenterPivotDataset(
        image_dir=train_image_dir, label_dir=train_label_dir)
    val_dataset = CenterPivotDataset(
        image_dir=val_image_dir, label_dir=val_label_dir)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True)
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)

    model = UNet3D(in_channels=in_channels, out_channels=out_channels,
                   inner_dims=inner_dims, encoder_type=encoder_type).cuda()

    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 'min', patience=5, factor=0.5)

    trainer = Trainer(model, train_loader, val_loader, criterion, optimizer,
                      scheduler, num_epochs, batch_size, best_model_path)

    trainer.train()


if __name__ == "__main__":
    main('configs/config.yaml')
