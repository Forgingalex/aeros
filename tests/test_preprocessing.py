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


def test_temporal_first_frame_is_warmup():
    """First temporal frame after reset should warm up with zero delta."""
    preprocessor = ImagePreprocessor()
    image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

    features = preprocessor.encode_temporal(image)

    assert features.warmup is True
    assert features.delta_model.shape == (224, 224)
    assert features.delta_vis.shape == (224, 224)
    assert features.delta_vis.dtype == np.uint8
    assert np.allclose(features.delta_model, 0.0)
    assert features.motion_energy == pytest.approx(0.0)


def test_temporal_zero_motion_for_repeated_frame():
    """Repeated identical frames should produce no temporal motion."""
    preprocessor = ImagePreprocessor()
    image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

    preprocessor.encode_temporal(image)
    features = preprocessor.encode_temporal(image)

    assert features.warmup is False
    assert np.allclose(features.delta_model, 0.0)
    assert features.motion_energy == pytest.approx(0.0)


def test_temporal_reset_restores_warmup():
    """Reset should clear temporal history for the next frame."""
    preprocessor = ImagePreprocessor()
    image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

    preprocessor.encode_temporal(image)
    preprocessor.reset()
    features = preprocessor.encode_temporal(image)

    assert features.warmup is True
    assert features.motion_energy == pytest.approx(0.0)


def test_temporal_motion_energy_detects_change():
    """Changed frames should create non-zero temporal contrast."""
    preprocessor = ImagePreprocessor()
    image_a = np.zeros((480, 640, 3), dtype=np.uint8)
    image_b = image_a.copy()
    image_b[120:280, 220:420] = 255

    preprocessor.encode_temporal(image_a)
    features = preprocessor.encode_temporal(image_b)

    assert features.warmup is False
    assert features.motion_energy > 0.0
    assert not np.allclose(features.delta_model, 0.0)
