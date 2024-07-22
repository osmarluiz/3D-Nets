
import torch
from torch.utils.data import DataLoader
import yaml

from datasets.center_pivot_dataset import CenterPivotDataset
from models.unet.unet_3d import UNet3D
from evaluator import compute_metrics
from evaluator import visualize_predictions

if __name__ == '__main__':
    with open('configs/config.yaml', 'r') as file:
        config = yaml.safe_load(file)

    test_image_dir = config['test']['image_dir']
    test_label_dir = config['test']['mask_dir']
    batch_size = config['test']['batch_size']
    num_workers = config['test']['num_workers']

    in_channels = config['model']['in_channels']
    out_channels = config['model']['out_channels']
    encoder_type = config['model']['encoder_type']
    inner_dims = config['model']['inner_dims']

    best_model_path = config['paths']['best_model_path']

    # Load the trained model
    model = UNet3D(
        in_channels=in_channels,
        out_channels=out_channels,
        inner_dims=inner_dims,
        encoder_type=encoder_type
    ).cuda()

    model.load_state_dict(torch.load(best_model_path))

    # Initialize test dataset and dataloader
    test_dataset = CenterPivotDataset(test_image_dir, test_label_dir)
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    # Compute metrics on the test dataloader
    test_metrics = compute_metrics(model, test_loader)
    print(f"Test Metrics:\n{test_metrics}")

    # Visualize predictions
    visualize_predictions(model, test_loader, num_images=5)
