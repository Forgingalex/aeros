"""Real-time inference engine."""

import torch
import numpy as np
import cv2
from typing import Optional, Union
from pathlib import Path
import time

# ONNX runtime optional: allows graceful degradation if not installed
# Enables edge deployment path without requiring PyTorch runtime
try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False
    ort = None


class InferenceEngine:
    """Handles real-time inference for heading estimation.
    
    Supports both PyTorch and ONNX models for deployment flexibility.
    """
    
    def __init__(
        self,
        model_path: Union[str, Path],
        use_onnx: bool = False,
        device: Optional[torch.device] = None,
    ):
        """Initialize inference engine.
        
        Args:
            model_path: Path to model checkpoint or ONNX file
            use_onnx: Whether to use ONNX runtime
            device: Device for PyTorch inference (ignored if use_onnx=True)
        """
        self.model_path = Path(model_path)
        self.use_onnx = use_onnx
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        
        if use_onnx:
            self._load_onnx_model()
        else:
            self._load_pytorch_model()
        
        self.inference_latencies = []
    
    def _load_pytorch_model(self) -> None:
        """Load PyTorch model."""
        from src.model.cnn import HeadingCNN
        
        self.model = HeadingCNN()
        checkpoint = torch.load(self.model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()
    
    def _load_onnx_model(self) -> None:
        """Load ONNX model."""
        if not ONNX_AVAILABLE:
            raise ImportError(
                "onnxruntime is not installed. Install it with: pip install onnxruntime"
            )
        self.session = ort.InferenceSession(
            str(self.model_path),
            providers=["CPUExecutionProvider", "CUDAExecutionProvider"],
        )
        self.input_name = self.session.get_inputs()[0].name
    
    def predict(self, image: np.ndarray) -> float:
        """Predict heading angle from image.
        
        Args:
            image: Preprocessed image (H, W, 3) in BGR format (will be converted to RGB)
            
        Returns:
            Predicted heading angle (radians)
        """
        start_time = time.time()
        
        # Critical fix: OpenCV VideoCapture returns BGR, but model was trained on RGB
        # Without this conversion, model receives inverted color channels, degrading accuracy
        if image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        if self.use_onnx:
            # ONNX inference path: optimized for edge deployment (lower memory, faster startup)
            image_nchw = np.transpose(image, (2, 0, 1)).astype(np.float32)
            image_batch = np.expand_dims(image_nchw, axis=0)
            image_batch = image_batch / 255.0
            
            outputs = self.session.run(None, {self.input_name: image_batch})
            prediction = float(outputs[0][0, 0])
        else:
            # PyTorch inference path: used during development, supports GPU acceleration
            image_tensor = torch.from_numpy(image).permute(2, 0, 1).float()
            image_tensor = image_tensor / 255.0
            image_batch = image_tensor.unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                output = self.model(image_batch)
                prediction = float(output[0, 0].cpu().item())
        
        inference_latency = time.time() - start_time
        self.inference_latencies.append(inference_latency)
        
        return prediction
    
    def get_fps(self) -> float:
        """Get average FPS from recent inferences.
        
        Returns:
            Average FPS
        """
        if not self.inference_latencies:
            return 0.0
        
        # Rolling window: last 100 inferences to smooth out transient spikes
        recent_latencies = self.inference_latencies[-100:]
        avg_latency = np.mean(recent_latencies)
        return 1.0 / avg_latency if avg_latency > 0 else 0.0
    
    def get_latency(self) -> float:
        """Get average latency from recent inferences.
        
        Returns:
            Average latency in milliseconds
        """
        if not self.inference_latencies:
            return 0.0
        
        recent_latencies = self.inference_latencies[-100:]
        return np.mean(recent_latencies) * 1000
    
    # TODO: Add perception confidence estimation (e.g., model uncertainty via Monte Carlo dropout
    # or ensemble variance) to enable confidence-conditioned control. Hypothesis: low-confidence
    # predictions indicate ambiguous scenes (e.g., featureless walls) where aggressive control
    # should be damped. Measure correlation between confidence and control stability in simulation.

