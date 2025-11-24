"""Tests for image preprocessing."""

import numpy as np
import pytest
from src.preprocessing.preprocessor import ImagePreprocessor


def test_preprocessor_initialization():
    """Test preprocessor initialization."""
    preprocessor = ImagePreprocessor()
    assert preprocessor.target_size == (224, 224)
    assert preprocessor.denoise is True
    assert preprocessor.normalize_brightness is True


def test_preprocess_image():
    """Test image preprocessing."""
    preprocessor = ImagePreprocessor()
    
    # Create dummy image
    image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    # Preprocess
    processed = preprocessor.preprocess(image)
    
    # Check output shape
    assert processed.shape == (224, 224, 3)
    assert processed.dtype == np.uint8


def test_preprocess_batch():
    """Test batch preprocessing."""
    preprocessor = ImagePreprocessor()
    
    # Create batch of images
    images = np.random.randint(0, 255, (4, 480, 640, 3), dtype=np.uint8)
    
    # Preprocess batch
    processed = preprocessor.preprocess_batch(images)
    
    # Check output shape
    assert processed.shape == (4, 224, 224, 3)

