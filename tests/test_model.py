"""Tests for model architecture."""

import torch
import pytest
from src.model.cnn import HeadingCNN


def test_model_initialization():
    """Test model initialization."""
    model = HeadingCNN()
    assert model is not None


def test_model_forward():
    """Test model forward pass."""
    model = HeadingCNN()
    model.eval()
    
    # Create dummy input
    batch_size = 4
    input_tensor = torch.randn(batch_size, 3, 224, 224)
    
    # Forward pass
    output = model(input_tensor)
    
    # Check output shape
    assert output.shape == (batch_size, 1)
    assert not torch.isnan(output).any()


def test_model_parameters():
    """Test model has trainable parameters."""
    model = HeadingCNN()
    num_params = sum(p.numel() for p in model.parameters())
    assert num_params > 0
    
    # Check model is reasonably sized (not too large)
    assert num_params < 10_000_000  # Less than 10M parameters

