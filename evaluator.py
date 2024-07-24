from typing import Dict, Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def calculate_iou(pred: torch.Tensor, target: torch.Tensor, threshold: float = 0.5) -> float:
    pred = (pred > threshold).float()
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection
    iou = intersection / union
    return iou.item()


def compute_metrics(model: nn.Module, dataloader: DataLoader) -> Dict[str, Any]:
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.cuda()
            labels = labels.cuda().float()

            outputs = model(images).squeeze(2)  # Squeeze the depth dimension
            outputs = torch.sigmoid(outputs)  # Apply sigmoid activation
            preds = (outputs > 0.5).float()

            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    # Flatten the last two dimensions (height and width) to avoid dimension mismatch
    all_preds = np.concatenate([p.reshape(p.shape[0], -1)
                               for p in all_preds], axis=0)
    all_labels = np.concatenate([l.reshape(l.shape[0], -1)
                                for l in all_labels], axis=0)

    accuracy = accuracy_score(all_labels.flatten(), all_preds.flatten())
    precision = precision_score(all_labels.flatten(), all_preds.flatten())
    recall = recall_score(all_labels.flatten(), all_preds.flatten())
    f1 = f1_score(all_labels.flatten(), all_preds.flatten())
    iou = calculate_iou(torch.tensor(all_preds).unsqueeze(
        1), torch.tensor(all_labels).unsqueeze(1))

    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'iou': iou
    }

    return metrics


def visualize_predictions(model: nn.Module, dataloader: DataLoader, num_images: int = 5):
    model.eval()
    images_so_far = 0

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.cuda()
            labels = labels.cuda().float()

            outputs = model(images).squeeze(2)  # Squeeze the depth dimension
            preds = (outputs > 0.5).float()  # Binarize predictions

            for i in range(images.size(0)):
                if images_so_far == num_images:
                    return

                image = images[i].cpu().numpy()
                label = labels[i].cpu().numpy()
                pred = preds[i].cpu().numpy()

                fig, axs = plt.subplots(1, 3, figsize=(15, 5))

                # Composite using the first three bands
                composite = np.stack(
                    [image[0, 0], image[1, 0], image[2, 0]], axis=-1)
                composite = (composite - composite.min()) / \
                    (composite.max() - composite.min())  # Normalize

                axs[0].imshow(composite)
                axs[0].set_title('Original Image (Composite)')
                axs[0].axis('off')

                axs[1].imshow(pred[0], cmap='gray')
                axs[1].set_title('Prediction')
                axs[1].axis('off')

                axs[2].imshow(label, cmap='gray')
                axs[2].set_title('Ground Truth')
                axs[2].axis('off')

                plt.show()

                images_so_far += 1
