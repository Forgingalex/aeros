"""FastAPI server for AEROS backend."""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import cv2
import numpy as np
import asyncio
from typing import Optional
from pathlib import Path
import json
import time
import os
import traceback

from src.utils.inference import InferenceEngine
from src.preprocessing.preprocessor import ImagePreprocessor
from src.control.pid_controller import PIDController


class PIDGains(BaseModel):
    """PID controller gains model."""
    kp: float
    ki: float
    kd: float

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
physics_loop_task: Optional[asyncio.Task] = None


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
    
    # Path traversal protection: restrict model loading to models/ directory only
    # Prevents loading arbitrary files via ../ sequences
    model_path_obj = Path(model_path)
    if model_path_obj.is_absolute():
        return JSONResponse(
            status_code=400,
            content={"status": "error", "message": "Model path must be relative"},
        )
    
    full_path = project_root / "models" / model_path_obj
    full_path = full_path.resolve()
    
    models_dir = (project_root / "models").resolve()
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
async def start_simulation(
    gui: Optional[bool] = None,
    enable_obstacles: bool = True,
    enable_turns: bool = True,
    lighting_variation: float = 0.3,
):
    """Start PyBullet simulation.
    
    Args:
        gui: Whether to show simulation GUI (auto-detects Docker if None)
        enable_obstacles: Whether to add obstacles in corridor
        enable_turns: Whether to add turns in corridor (future feature)
        lighting_variation: Lighting variation (0-1)
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
    
    # Auto-detect headless environment (Docker/CI) to prevent GUI initialization failures
    if gui is None:
        gui = os.getenv("DISPLAY") is not None and os.getenv("DISPLAY") != ""
    
    # Clamp lighting variation
    lighting_variation = max(0.0, min(1.0, lighting_variation))
    
    try:
        simulation = DroneSimulation(
            gui=gui,
            enable_obstacles=enable_obstacles,
            enable_turns=enable_turns,
            lighting_variation=lighting_variation,
        )
        use_simulation = True
        return {
            "status": "success",
            "message": f"Simulation started (gui={gui}, obstacles={enable_obstacles}, lighting={lighting_variation})",
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


@app.post("/set_pid_gains")
async def set_pid_gains(gains: PIDGains):
    """Configure PID controller gains dynamically.
    
    Args:
        gains: PIDGains model with kp, ki, kd values
    """
    global pid_controller
    
    if pid_controller is None:
        return JSONResponse(
            status_code=400,
            content={
                "status": "error",
                "message": "PID controller not initialized",
            },
        )
    
    # Update PID controller gains
    pid_controller.kp = gains.kp
    pid_controller.ki = gains.ki
    pid_controller.kd = gains.kd
    
    print(f"PID gains updated: Kp={gains.kp}, Ki={gains.ki}, Kd={gains.kd}")
    
    return {
        "status": "success",
        "message": "PID gains updated successfully",
        "gains": {
            "kp": gains.kp,
            "ki": gains.ki,
            "kd": gains.kd,
        },
    }


@app.get("/get_pid_gains")
async def get_pid_gains():
    """Get current PID controller gains.
    
    Returns:
        Current kp, ki, kd values
    """
    global pid_controller
    
    if pid_controller is None:
        return JSONResponse(
            status_code=400,
            content={
                "status": "error",
                "message": "PID controller not initialized",
            },
        )
    
    return {
        "status": "success",
        "gains": {
            "kp": pid_controller.kp,
            "ki": pid_controller.ki,
            "kd": pid_controller.kd,
        },
    }


async def physics_loop():
    """Background task running at fixed 60 Hz independent of WebSocket client count.
    
    Prevents physics time warp: without this, multiple clients would cause
    simulation.step() to be called multiple times per frame, speeding up physics.
    """
    global simulation, use_simulation
    while use_simulation and simulation:
        try:
            simulation.step()
            await asyncio.sleep(1.0 / 60.0)  # Fixed 60 Hz physics timestep
        except Exception as e:
            print(f"Physics loop error: {e}")
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
    global physics_loop_task, use_simulation, simulation
    
    print(f"Incoming WebSocket connection attempt from {websocket.client}")
    try:
        await websocket.accept()
        print("WebSocket connection accepted successfully")
    except Exception as e:
        print(f"ERROR: Failed to accept WebSocket connection: {e}")
        return
    
    # Start background physics loop only once (shared across all WebSocket connections)
    # Prevents multiple physics loops from running simultaneously
    if use_simulation and simulation and (physics_loop_task is None or physics_loop_task.done()):
        physics_loop_task = asyncio.create_task(physics_loop())
        print("Physics loop started")
    
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
    
    # Initialize video capture (webcam or simulation)
    video_capture = None
    if use_simulation and simulation:
        print("Using simulation camera")
        video_capture = None  # Use simulation camera
    else:
        # Try to open webcam
        print("Attempting to open camera (index 0)...")
        video_capture = cv2.VideoCapture(0)
        if not video_capture.isOpened():
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
    
    # Comprehensive error handling for entire streaming loop
    try:
        frame_count = 0
        start_time = time.time()
        
        while True:
            try:
                if use_simulation and simulation:
                    try:
                        frame = simulation.get_camera_image()
                        # Convert RGB to BGR for OpenCV compatibility
                        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    except Exception as e:
                        print(f"ERROR: Failed to get simulation frame: {e}")
                        print(f"Traceback: {traceback.format_exc()}")
                        break
                else:
                    try:
                        ret, frame = video_capture.read()
                        if not ret:
                            print("WARNING: Failed to read frame from camera")
                            break
                    except Exception as e:
                        print(f"ERROR: Camera read failed: {e}")
                        print(f"Traceback: {traceback.format_exc()}")
                        break
                
                # Preprocess frame for perception pipeline
                try:
                    preprocessed_frame = preprocessor.preprocess(frame)
                except Exception as e:
                    print(f"ERROR: Preprocessing failed: {e}")
                    print(f"Traceback: {traceback.format_exc()}")
                    # Non-critical error: skip frame to maintain stream continuity
                    await asyncio.sleep(1.0 / 30.0)
                    continue
                
                # Predict heading from preprocessed frame
                try:
                    heading = inference_engine.predict(preprocessed_frame)
                except Exception as e:
                    print(f"ERROR: Inference failed: {e}")
                    print(f"Traceback: {traceback.format_exc()}")
                    # Non-critical error: skip frame to maintain stream continuity
                    await asyncio.sleep(1.0 / 30.0)
                    continue
                
                # Compute control command from heading error
                try:
                    angular_velocity_command = pid_controller.compute_control(heading)
                except Exception as e:
                    print(f"ERROR: PID control computation failed: {e}")
                    print(f"Traceback: {traceback.format_exc()}")
                    # Fail-safe: zero command prevents runaway control
                    angular_velocity_command = 0.0
                
                try:
                    fps = inference_engine.get_fps()
                    latency = inference_engine.get_latency()
                except Exception as e:
                    print(f"WARNING: Failed to get metrics: {e}")
                    fps = 0.0
                    latency = 0.0
                
                drone_state = None
                if use_simulation and simulation:
                    try:
                        drone_state = simulation.get_drone_state()
                        # Control applied here; physics stepping happens in background task at 60 Hz
                        simulation.apply_control(angular_velocity_command)
                    except Exception as e:
                        print(f"WARNING: Simulation state/control failed: {e}")
                        print(f"Traceback: {traceback.format_exc()}")
                
                # Encode frame as JPEG (quality 80 balances size vs visual quality for real-time streaming)
                try:
                    _, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                    encoded_frame_bytes = buffer.tobytes()
                except Exception as e:
                    print(f"ERROR: Frame encoding failed: {e}")
                    print(f"Traceback: {traceback.format_exc()}")
                    # Non-critical error: skip frame to maintain stream continuity
                    await asyncio.sleep(1.0 / 30.0)
                    continue
                
                try:
                    await websocket.send_json({
                        "type": "telemetry",
                        "heading": float(heading),
                        "control_output": float(angular_velocity_command),
                        "fps": float(fps),
                        "latency_ms": float(latency),
                        "frame_count": frame_count,
                        "drone_state": drone_state,
                    })
                except Exception as e:
                    print(f"ERROR: Failed to send telemetry: {e}")
                    print(f"Traceback: {traceback.format_exc()}")
                    # Connection broken: exit loop to allow reconnection
                    break
                
                try:
                    await websocket.send_bytes(encoded_frame_bytes)
                except Exception as e:
                    print(f"ERROR: Failed to send frame: {e}")
                    print(f"Traceback: {traceback.format_exc()}")
                    # Connection broken: exit loop to allow reconnection
                    break
                
                frame_count += 1
                
                # Throttle to 30 FPS to match frontend display rate and reduce network bandwidth
                await asyncio.sleep(1.0 / 30.0)
                
            except WebSocketDisconnect:
                print("WebSocket disconnected (client closed)")
                break
            except Exception as e:
                print(f"ERROR: Unexpected error in frame loop: {e}")
                print(f"Traceback: {traceback.format_exc()}")
                # Try to send error to client
                try:
                    await websocket.send_json({"error": str(e), "type": "error"})
                except:
                    pass  # Connection might already be closed
                # Break on unexpected errors
                break
                
    except WebSocketDisconnect:
        print("WebSocket disconnected (client closed)")
    except Exception as e:
        print(f"CRITICAL ERROR: WebSocket stream error: {e}")
        print(f"Full traceback: {traceback.format_exc()}")
        try:
            await websocket.send_json({"error": str(e), "type": "critical_error"})
        except:
            pass  # Connection might already be closed
    finally:
        print("Cleaning up WebSocket connection...")
        if video_capture and not use_simulation:
            video_capture.release()
            print("Camera released")
        try:
            await websocket.close()
            print("WebSocket closed")
        except:
            pass  # Might already be closed


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

