"""Standalone training script for AEROS heading estimation model."""

import sys
from pathlib import Path
import json
import numpy as np
import cv2
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model.cnn import HeadingCNN
from src.model.trainer import ModelTrainer


class CorridorDataset(Dataset):
    """Dataset for corridor images and heading angles."""
    
    def __init__(self, metadata_path, data_root, transform=None):
        """Initialize dataset.
        
        Args:
            metadata_path: Path to metadata JSON file
            data_root: Root directory for images
            transform: Optional image transforms
        """
        with open(metadata_path, 'r') as f:
            self.metadata = json.load(f)
        
        self.data_root = Path(data_root)
        self.transform = transform
    
    def __len__(self):
        return len(self.metadata)
    
    def __getitem__(self, idx):
        item = self.metadata[idx]
        
        # Load image
        image_path = self.data_root / item['image_path']
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Convert to tensor
        image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        
        if self.transform:
            image = self.transform(image)
        
        # Get heading
        heading = torch.tensor([item['heading']], dtype=torch.float32)
        
        return image, heading


def main():
    """Main training function."""
    print("=" * 60)
    print("AEROS Model Training")
    print("=" * 60)
    
    # Setup paths
    data_root = Path("data/synthetic")
    checkpoint_dir = Path("models/checkpoints")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Check if data exists
    train_metadata = data_root / "train_metadata.json"
    val_metadata = data_root / "val_metadata.json"
    
    if not train_metadata.exists() or not val_metadata.exists():
        print(f"Error: Training data not found in {data_root}")
        print("Please run: python scripts/generate_synthetic_data.py")
        return
    
    # Load datasets
    print("\nLoading datasets...")
    train_dataset = CorridorDataset(
        metadata_path=train_metadata,
        data_root=data_root,
    )
    val_dataset = CorridorDataset(
        metadata_path=val_metadata,
        data_root=data_root,
    )
    
    train_loader = DataLoader(
        train_dataset, batch_size=32, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(
        val_dataset, batch_size=32, shuffle=False, num_workers=0
    )
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    
    # Initialize model
    print("\nInitializing model...")
    model = HeadingCNN()
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")
    
    # Initialize trainer
    print("\nInitializing trainer...")
    trainer = ModelTrainer(
        model=model,
        learning_rate=1e-3,
        weight_decay=1e-4,
    )
    
    # Train
    print("\nStarting training...")
    print("-" * 60)
    
    # Use fewer epochs for quick testing (change to 50 for full training)
    num_epochs = 5  # Quick test - change to 50 for full training
    history = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=num_epochs,
        checkpoint_dir=checkpoint_dir,
        save_best=True,
    )
    
    print("\n" + "=" * 60)
    print("Training completed!")
    print(f"Best model saved to: {checkpoint_dir / 'best_model.pth'}")
    print("=" * 60)
    
    # Print final metrics
    print("\nFinal Training Metrics:")
    print(f"  Final train loss: {history['train_loss'][-1]:.4f}")
    print(f"  Final val loss: {history['val_loss'][-1]:.4f}")
    print(f"  Best val loss: {min(history['val_loss']):.4f}")


if __name__ == "__main__":
    main()

