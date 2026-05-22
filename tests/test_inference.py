"""Tests for inference runtime helpers."""

import numpy as np

from src.utils.inference import PassthroughInferenceEngine


def test_passthrough_inference_engine_returns_neutral_prediction():
    """Passthrough mode should keep the stream alive with neutral control."""
    engine = PassthroughInferenceEngine()
    image = np.zeros((224, 224, 3), dtype=np.uint8)

    assert engine.predict(image) == 0.0
    assert engine.get_fps() == 0.0
    assert engine.get_latency() == 0.0
    assert engine.is_passthrough is True
