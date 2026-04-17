"""Training loop for neural models (text encoder & multimodal)."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        loss_fn: nn.Module,
        device: torch.device,
        output_dir: str = "outputs/",
        grad_clip: float = 1.0,
    ) -> None:
        self.model = model.to(device)
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.device = device
        self.output_dir = Path(output_dir)
        self.grad_clip = grad_clip
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def train_epoch(self, loader: DataLoader) -> float:
        """Run one training epoch, return mean loss."""
        # TODO: implement training step
        raise NotImplementedError

    def eval_epoch(self, loader: DataLoader) -> dict:
        """Evaluate on a DataLoader, return metrics dict."""
        # TODO: implement evaluation step
        raise NotImplementedError

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int,
        patience: int = 10,
    ) -> None:
        """Train with early stopping on validation loss."""
        # TODO: implement full training loop with early stopping + checkpointing
        raise NotImplementedError

    def save_checkpoint(self, epoch: int, val_loss: float) -> None:
        path = self.output_dir / f"checkpoint_epoch{epoch:03d}_val{val_loss:.4f}.pt"
        torch.save({
            "epoch": epoch,
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "val_loss": val_loss,
        }, path)
