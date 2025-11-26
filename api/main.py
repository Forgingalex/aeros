"""FastAPI server for AEROS backend."""

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import cv2
import numpy as np
import asyncio
from typing import Optional
from pathlib import Path
import json
import time
import os

from src.utils.inference import InferenceEngine
from src.preprocessing.preprocessor import ImagePreprocessor
from src.control.pid_controller import PIDController

# PyBullet simulation is optional
try:
    from src.simulation.drone_sim import DroneSimulation
    SIMULATION_AVAILABLE = True
except ImportError:
    SIMULATION_AVAILABLE = False
    # Create a dummy class for type hints
    class DroneSimulation:
        pass

app = FastAPI(title="AEROS API", version="1.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state
inference_engine: Optional[InferenceEngine] = None
preprocessor: Optional[ImagePreprocessor] = None
pid_controller: Optional[PIDController] = None
simulation: Optional[DroneSimulation] = None
use_simulation: bool = False
simulation_task: Optional[asyncio.Task] = None


@app.on_event("startup")
async def startup():
    """Initialize components on startup."""
    global inference_engine, preprocessor, pid_controller
    
    # Initialize preprocessor
    preprocessor = ImagePreprocessor()
    
    # Initialize PID controller
    pid_controller = PIDController(kp=1.0, ki=0.1, kd=0.5)
    
    print("AEROS API started")


@app.on_event("shutdown")
async def shutdown():
    """Cleanup on shutdown."""
    global simulation
    if simulation:
        simulation.close()


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "AEROS API",
        "status": "running",
        "version": "1.0.0",
    }


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy"}


@app.post("/load_model")
async def load_model(model_path: str, use_onnx: bool = False):
    """Load inference model.
    
    Args:
        model_path: Path to model checkpoint or ONNX file (relative to models/)
        use_onnx: Whether model is ONNX format
    """
    global inference_engine
    
    # Security: Only allow files from models/ directory
    model_path_obj = Path(model_path)
    if model_path_obj.is_absolute():
        return JSONResponse(
            status_code=400,
            content={"status": "error", "message": "Model path must be relative"},
        )
    
    # Resolve to models directory
    full_path = Path("models") / model_path_obj
    full_path = full_path.resolve()
    
    # Ensure it's within models directory (prevent path traversal)
    models_dir = Path("models").resolve()
    if not str(full_path).startswith(str(models_dir)):
        return JSONResponse(
            status_code=403,
            content={"status": "error", "message": "Model path must be within models/ directory"},
        )
    
    if not full_path.exists():
        return JSONResponse(
            status_code=404,
            content={"status": "error", "message": f"Model file not found: {model_path}"},
        )
    
    try:
        inference_engine = InferenceEngine(
            model_path=str(full_path),
            use_onnx=use_onnx,
        )
        return {
            "status": "success",
            "message": f"Model loaded from {model_path}",
            "use_onnx": use_onnx,
        }
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": str(e)},
        )


@app.post("/start_simulation")
async def start_simulation(gui: Optional[bool] = None):
    """Start PyBullet simulation.
    
    Args:
        gui: Whether to show simulation GUI (auto-detects Docker if None)
    """
    global simulation, use_simulation
    
    if not SIMULATION_AVAILABLE:
        return JSONResponse(
            status_code=503,
            content={
                "status": "error",
                "message": "PyBullet is not installed. Install it with: pip install pybullet",
            },
        )
    
    # Auto-detect Docker/headless mode
    if gui is None:
        gui = os.getenv("DISPLAY") is not None and os.getenv("DISPLAY") != ""
    
    try:
        simulation = DroneSimulation(gui=gui)
        use_simulation = True
        return {
            "status": "success",
            "message": f"Simulation started (gui={gui})",
        }
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": str(e)},
        )


@app.post("/stop_simulation")
async def stop_simulation():
    """Stop PyBullet simulation."""
    global simulation, use_simulation
    
    if simulation:
        simulation.close()
        simulation = None
        use_simulation = False
    
    return {"status": "success", "message": "Simulation stopped"}


async def simulation_loop():
    """Background task to step simulation independently."""
    global simulation, use_simulation
    while use_simulation and simulation:
        try:
            simulation.step()
            await asyncio.sleep(1.0 / 60.0)  # 60 Hz physics
        except Exception as e:
            print(f"Simulation loop error: {e}")
            break


@app.websocket("/ws/test")
async def websocket_test(websocket: WebSocket):
    """Simple WebSocket test endpoint."""
    print("Test WebSocket connection attempt")
    try:
        await websocket.accept()
        print("Test WebSocket accepted")
        await websocket.send_json({"message": "WebSocket test successful"})
        await websocket.close()
    except Exception as e:
        print(f"Test WebSocket error: {e}")
        import traceback
        traceback.print_exc()


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time streaming.
    
    Streams:
    - Camera frames
    - Predicted heading
    - Controller output
    - FPS and latency
    """
    global simulation_task, use_simulation, simulation
    
    print(f"Incoming WebSocket connection attempt from {websocket.client}")
    try:
        await websocket.accept()
        print("WebSocket connection accepted successfully")
    except Exception as e:
        print(f"ERROR: Failed to accept WebSocket connection: {e}")
        return
    
    # Start simulation loop if using simulation and not already running
    if use_simulation and simulation and (simulation_task is None or simulation_task.done()):
        simulation_task = asyncio.create_task(simulation_loop())
        print("Simulation loop started")
    
    if not inference_engine:
        error_msg = "Model not loaded. Call /load_model first."
        print(f"ERROR: {error_msg}")
        try:
            await websocket.send_json({
                "error": error_msg,
            })
            await websocket.close()
        except Exception as e:
            print(f"Error sending error message: {e}")
        return
    
    # Initialize camera (webcam or simulation)
    camera = None
    if use_simulation and simulation:
        print("Using simulation camera")
        camera = None  # Use simulation camera
    else:
        # Try to open webcam
        print("Attempting to open camera (index 0)...")
        camera = cv2.VideoCapture(0)
        if not camera.isOpened():
            error_msg = "Could not open camera. Check if camera is available and not in use by another application."
            print(f"ERROR: {error_msg}")
            print("Trying to get camera backend info...")
            print(f"OpenCV version: {cv2.__version__}")
            await websocket.send_json({
                "error": error_msg,
            })
            await websocket.close()
            return
        print("Camera opened successfully")
    
    print("WebSocket connection established - starting stream")
    
    try:
        frame_count = 0
        start_time = time.time()
        
        while True:
            # Get frame
            if use_simulation and simulation:
                frame = simulation.get_camera_image()
                # Convert RGB to BGR for OpenCV compatibility
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            else:
                ret, frame = camera.read()
                if not ret:
                    print("WARNING: Failed to read frame from camera")
                    break
            
            # Preprocess
            processed = preprocessor.preprocess(frame)
            
            # Predict heading
            heading = inference_engine.predict(processed)
            
            # Compute control output
            control_output = pid_controller.update(heading)
            
            # Get metrics
            fps = inference_engine.get_fps()
            latency = inference_engine.get_latency()
            
            # Get drone state if simulation
            drone_state = None
            if use_simulation and simulation:
                drone_state = simulation.get_drone_state()
                # Apply control (simulation.step() runs in background task)
                simulation.apply_control(control_output)
            
            # Encode frame as base64
            _, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            frame_base64 = buffer.tobytes()
            
            # Send telemetry
            await websocket.send_json({
                "type": "telemetry",
                "heading": float(heading),
                "control_output": float(control_output),
                "fps": float(fps),
                "latency_ms": float(latency),
                "frame_count": frame_count,
                "drone_state": drone_state,
            })
            
            # Send frame
            await websocket.send_bytes(frame_base64)
            
            frame_count += 1
            
            # Small delay to control frame rate
            await asyncio.sleep(1.0 / 30.0)  # Target 30 FPS
            
    except WebSocketDisconnect:
        print("WebSocket disconnected (client closed)")
    except Exception as e:
        import traceback
        print(f"WebSocket error: {e}")
        print(f"Traceback: {traceback.format_exc()}")
        try:
            await websocket.send_json({"error": str(e)})
        except:
            pass  # Connection might already be closed
    finally:
        print("Cleaning up WebSocket connection...")
        if camera and not use_simulation:
            camera.release()
            print("Camera released")
        try:
            await websocket.close()
            print("WebSocket closed")
        except:
            pass  # Might already be closed


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

