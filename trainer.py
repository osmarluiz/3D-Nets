from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from torch.optim.lr_scheduler import ReduceLROnPlateau


class Trainer:
    def __init__(self, model: nn.Module, train_loader: DataLoader, val_loader: DataLoader,
                 criterion: nn.Module, optimizer: Optimizer, scheduler: ReduceLROnPlateau,
                 num_epochs: int, batch_size: int, best_model_path: str):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.scaler = torch.cuda.amp.GradScaler()
        self.best_val_loss = float('inf')
        self.best_model_path = best_model_path

    def calculate_iou(self, pred: torch.Tensor, target: torch.Tensor,
                      threshold: float = 0.5) -> float:
        pred = (pred > threshold).float()
        target = target.float()
        intersection = (pred * target).sum(dim=(1, 2, 3))
        union = pred.sum(dim=(1, 2, 3)) + \
            target.sum(dim=(1, 2, 3)) - intersection
        iou = (intersection + 1e-6) / (union + 1e-6)
        return iou.mean().item()

    def train(self):
        for epoch in range(self.num_epochs):
            self.model.train()
            running_loss = 0.0
            train_iou = 0.0
            train_steps = 0

            with tqdm(total=len(self.train_loader), desc=f'Epoch {epoch + 1}/{self.num_epochs}', unit='batch') as pbar:
                for images, labels in self.train_loader:
                    images: torch.Tensor = images.cuda()
                    labels: torch.Tensor = labels.cuda().float()

                    self.optimizer.zero_grad()
                    with torch.cuda.amp.autocast():
                        outputs = self.model(images)
                        outputs = torch.sigmoid(outputs.squeeze(2))
                        loss = self.criterion(outputs, labels.unsqueeze(1))

                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()

                    running_loss += loss.item() * images.size(0)
                    train_iou += self.calculate_iou(outputs,
                                                    labels.unsqueeze(1))
                    train_steps += 1

                    pbar.set_postfix(
                        {'loss': loss.item(), 'iou': train_iou / train_steps})
                    pbar.update(1)

            epoch_loss = running_loss / \
                len(self.train_loader.dataset)  # type: ignore
            train_iou /= train_steps

            val_loss, val_iou = self.validate()

            self.scheduler.step(val_loss)

            print(f'Epoch {epoch + 1}/{self.num_epochs}, Train Loss: {epoch_loss:.4f}, Train IoU: {train_iou:.4f}, Val Loss: {val_loss:.4f}, Val IoU: {val_iou:.4f}')

            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                torch.save(self.model.state_dict(), self.best_model_path)
                print(
                    f'Saved Best Model at Epoch {epoch + 1} with Validation Loss: {val_loss:.4f}')

        print('Training Completed.')

    def validate(self) -> tuple[float, float]:
        self.model.eval()
        val_loss = 0.0
        val_iou = 0.0
        val_steps = 0

        with torch.no_grad():
            for val_images, val_labels in self.val_loader:
                val_images = val_images.cuda()
                val_labels = val_labels.cuda().float()

                with torch.cuda.amp.autocast():
                    val_outputs = self.model(val_images)
                    val_outputs = torch.sigmoid(val_outputs.squeeze(2))
                    val_loss += self.criterion(val_outputs,
                                               val_labels.unsqueeze(1)).item() * val_images.size(0)
                    val_iou += self.calculate_iou(val_outputs,
                                                  val_labels.unsqueeze(1))
                    val_steps += 1

        val_loss /= len(self.val_loader.dataset)  # type: ignore
        val_iou /= val_steps

        return val_loss, val_iou
