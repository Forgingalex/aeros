"""Image preprocessing pipeline for heading estimation."""

from dataclasses import dataclass
from typing import Tuple

import cv2
import numpy as np


@dataclass
class TemporalFrameFeatures:
    """Temporal-contrast features derived from the latest frame."""

    delta_model: np.ndarray
    delta_vis: np.ndarray
    motion_energy: float
    warmup: bool


class ImagePreprocessor:
    """Preprocesses images for model inference.
    
    Performs resize, denoising, and brightness normalization.
    """
    
    def __init__(
        self,
        target_size: Tuple[int, int] = (224, 224),
        denoise: bool = True,
        normalize_brightness: bool = True,
    ):
        """Initialize preprocessor.
        
        Args:
            target_size: Target image dimensions (width, height)
            denoise: Whether to apply denoising
            normalize_brightness: Whether to normalize brightness
        """
        self.target_size = target_size
        self.denoise = denoise
        self.normalize_brightness = normalize_brightness
        self._clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        self._previous_temporal_frame: np.ndarray | None = None

    def reset(self) -> None:
        """Clear any stateful preprocessing history."""
        self._previous_temporal_frame = None

    def _resize_image(self, image: np.ndarray) -> np.ndarray:
        """Resize incoming images to the model target resolution."""
        return cv2.resize(image, self.target_size, interpolation=cv2.INTER_LINEAR)

    def _apply_legacy_denoise(self, image: np.ndarray) -> np.ndarray:
        """Apply the low-latency denoising path used by the baseline model."""
        if not self.denoise:
            return image

        # Performance tradeoff: Gaussian blur instead of Non-Local Means (NLM)
        # NLM provides better denoising but costs 50-200ms per frame, breaking real-time constraint
        # Gaussian blur keeps preprocessing under 10ms while providing sufficient noise reduction
        return cv2.GaussianBlur(image, (5, 5), 0)

    def _apply_brightness_normalization(self, image: np.ndarray) -> np.ndarray:
        """Normalize luminance while preserving the original color channels."""
        if not self.normalize_brightness:
            return image

        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)
        l_channel = self._clahe.apply(l_channel)
        normalized = cv2.merge([l_channel, a_channel, b_channel])
        return cv2.cvtColor(normalized, cv2.COLOR_LAB2BGR)

    def _prepare_temporal_frame(self, image: np.ndarray) -> np.ndarray:
        """Prepare a normalized grayscale frame for temporal differencing."""
        resized = self._resize_image(image)
        grayscale = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(grayscale, (5, 5), 0)
        return blurred.astype(np.float32) / 255.0

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """Preprocess a single image.
        
        Args:
            image: Input image (BGR format from OpenCV)
            
        Returns:
            Preprocessed image ready for model input
        """
        processed = self._resize_image(image)
        processed = self._apply_legacy_denoise(processed)
        processed = self._apply_brightness_normalization(processed)
        return processed

    def encode_temporal(self, image: np.ndarray) -> TemporalFrameFeatures:
        """Encode temporal contrast from the current frame.

        The first frame after a reset warms up the temporal state and emits a
        zero delta field so downstream consumers can distinguish cold-start
        behavior from true low-motion scenes.
        """
        current_frame = self._prepare_temporal_frame(image)
        warmup = self._previous_temporal_frame is None

        if warmup:
            delta_model = np.zeros_like(current_frame, dtype=np.float32)
        else:
            delta_model = np.clip(
                current_frame - self._previous_temporal_frame,
                -1.0,
                1.0,
            ).astype(np.float32)

        self._previous_temporal_frame = current_frame

        motion_energy = float(np.mean(np.abs(delta_model)))
        delta_vis = np.clip((delta_model + 1.0) * 127.5, 0, 255).astype(np.uint8)

        return TemporalFrameFeatures(
            delta_model=delta_model,
            delta_vis=delta_vis,
            motion_energy=motion_energy,
            warmup=warmup,
        )
    
    # TODO: Explore event-based or sparse visual representations for ultra-low latency perception.
    # Hypothesis: Converting frames to event streams (pixel-level brightness changes) could reduce
    # preprocessing to <2ms while maintaining heading estimation accuracy. Test with DVS camera
    # simulation or frame-to-event conversion, measure latency vs accuracy tradeoff on Jetson Nano.
    
    def preprocess_batch(self, images: np.ndarray) -> np.ndarray:
        """Preprocess a batch of images.
        
        Args:
            images: Array of images (N, H, W, C)
            
        Returns:
            Preprocessed batch
        """
        return np.array([self.preprocess(img) for img in images])
