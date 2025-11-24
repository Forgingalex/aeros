"""Training utilities for heading estimation model."""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Optional, Dict, Any
import os
from pathlib import Path


class ModelTrainer:
    """Handles model training, validation, and checkpointing."""
    
    def __init__(
        self,
        model: nn.Module,
        device: Optional[torch.device] = None,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
    ):
        """Initialize trainer.
        
        Args:
            model: PyTorch model to train
            device: Device to train on (cuda/cpu)
            learning_rate: Learning rate for optimizer
            weight_decay: Weight decay for regularization
        """
        self.model = model
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model.to(self.device)
        
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(
            self.model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.5, patience=5
        )
        
    def train_epoch(
        self, train_loader: DataLoader, epoch: int
    ) -> Dict[str, float]:
        """Train for one epoch.
        
        Args:
            train_loader: Training data loader
            epoch: Current epoch number
            
        Returns:
            Dictionary with training metrics
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        for batch_idx, (images, targets) in enumerate(train_loader):
            images = images.to(self.device)
            targets = targets.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            if batch_idx % 10 == 0:
                print(
                    f"Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}"
                )
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        return {"loss": avg_loss}
    
    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        """Validate model.
        
        Args:
            val_loader: Validation data loader
            
        Returns:
            Dictionary with validation metrics
        """
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for images, targets in val_loader:
                images = images.to(self.device)
                targets = targets.to(self.device)
                
                outputs = self.model(images)
                loss = self.criterion(outputs, targets)
                
                total_loss += loss.item()
                num_batches += 1
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        return {"loss": avg_loss}
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        num_epochs: int,
        checkpoint_dir: Optional[Path] = None,
        save_best: bool = True,
    ) -> Dict[str, Any]:
        """Train model for multiple epochs.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            num_epochs: Number of epochs to train
            checkpoint_dir: Directory to save checkpoints
            save_best: Whether to save best model based on validation loss
            
        Returns:
            Training history dictionary
        """
        if checkpoint_dir:
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        best_val_loss = float("inf")
        history = {"train_loss": [], "val_loss": []}
        
        for epoch in range(1, num_epochs + 1):
            print(f"\nEpoch {epoch}/{num_epochs}")
            print("-" * 50)
            
            # Train
            train_metrics = self.train_epoch(train_loader, epoch)
            history["train_loss"].append(train_metrics["loss"])
            
            # Validate
            val_metrics = self.validate(val_loader)
            history["val_loss"].append(val_metrics["loss"])
            
            # Learning rate scheduling
            self.scheduler.step(val_metrics["loss"])
            
            print(f"Train Loss: {train_metrics['loss']:.4f}")
            print(f"Val Loss: {val_metrics['loss']:.4f}")
            
            # Save checkpoint
            if checkpoint_dir:
                checkpoint_path = checkpoint_dir / f"checkpoint_epoch_{epoch}.pth"
                self.save_checkpoint(checkpoint_path, epoch, val_metrics["loss"])
                
                if save_best and val_metrics["loss"] < best_val_loss:
                    best_val_loss = val_metrics["loss"]
                    best_path = checkpoint_dir / "best_model.pth"
                    self.save_checkpoint(best_path, epoch, val_metrics["loss"])
                    print(f"Saved best model (val_loss: {best_val_loss:.4f})")
        
        return history
    
    def save_checkpoint(
        self, path: Path, epoch: int, val_loss: float
    ) -> None:
        """Save model checkpoint.
        
        Args:
            path: Path to save checkpoint
            epoch: Current epoch
            val_loss: Validation loss
        """
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "val_loss": val_loss,
            },
            path,
        )
    
    def load_checkpoint(self, path: Path) -> None:
        """Load model checkpoint.
        
        Args:
            path: Path to checkpoint file
        """
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        return checkpoint.get("epoch", 0)

