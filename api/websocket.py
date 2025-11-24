"""WebSocket utilities for real-time streaming."""

import asyncio
import json
from typing import Optional, Callable
import cv2
import numpy as np


class TelemetryStreamer:
    """Handles WebSocket telemetry streaming."""
    
    def __init__(self, websocket):
        """Initialize streamer.
        
        Args:
            websocket: WebSocket connection
        """
        self.websocket = websocket
        self.running = False
    
    async def send_telemetry(
        self,
        heading: float,
        control_output: float,
        fps: float,
        latency_ms: float,
        frame_count: int,
        drone_state: Optional[dict] = None,
    ):
        """Send telemetry data.
        
        Args:
            heading: Predicted heading angle
            control_output: PID controller output
            fps: Current FPS
            latency_ms: Inference latency in milliseconds
            frame_count: Frame counter
            drone_state: Optional drone state dictionary
        """
        await self.websocket.send_json({
            "type": "telemetry",
            "heading": float(heading),
            "control_output": float(control_output),
            "fps": float(fps),
            "latency_ms": float(latency_ms),
            "frame_count": frame_count,
            "drone_state": drone_state,
        })
    
    async def send_frame(self, frame: np.ndarray, quality: int = 80):
        """Send frame as JPEG.
        
        Args:
            frame: Image frame (BGR format)
            quality: JPEG quality (0-100)
        """
        _, buffer = cv2.imencode(
            ".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality]
        )
        await self.websocket.send_bytes(buffer.tobytes())

