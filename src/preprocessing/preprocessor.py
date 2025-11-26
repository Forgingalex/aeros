"""Image preprocessing pipeline for heading estimation."""

import cv2
import numpy as np
from typing import Tuple, Optional


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
    
    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """Preprocess a single image.
        
        Args:
            image: Input image (BGR format from OpenCV)
            
        Returns:
            Preprocessed image ready for model input
        """
        # Resize
        processed = cv2.resize(image, self.target_size, interpolation=cv2.INTER_LINEAR)
        
        # Denoise - use fast Gaussian blur instead of expensive NLM
        if self.denoise:
            # Use Gaussian blur for real-time performance
            # NLM is too slow (50-200ms per frame)
            processed = cv2.GaussianBlur(processed, (5, 5), 0)
        
        # Brightness normalization
        if self.normalize_brightness:
            lab = cv2.cvtColor(processed, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            l = clahe.apply(l)
            processed = cv2.merge([l, a, b])
            processed = cv2.cvtColor(processed, cv2.COLOR_LAB2BGR)
        
        return processed
    
    def preprocess_batch(self, images: np.ndarray) -> np.ndarray:
        """Preprocess a batch of images.
        
        Args:
            images: Array of images (N, H, W, C)
            
        Returns:
            Preprocessed batch
        """
        return np.array([self.preprocess(img) for img in images])

