"""FastAPI server for AEROS backend."""

import asyncio
import os
import sys
import traceback
from pathlib import Path
from typing import Literal, Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from api.runtime import AuthoritativeStreamRuntime
from src.control.pid_controller import PIDController
from src.preprocessing.preprocessor import ImagePreprocessor
from src.utils.benchmark import BenchmarkRecorder
from src.utils.inference import InferenceEngine, PassthroughInferenceEngine


class PIDGains(BaseModel):
    """PID controller gains model."""

    kp: float
    ki: float
    kd: float


class RuntimeConfig(BaseModel):
    """Runtime mode configuration surface for the authoritative loop."""

    runtime_mode: Literal["legacy", "omega_shadow", "omega"] = "legacy"
    safety_mode: Literal["legacy", "cga_shadow", "cga_advisory"] = "legacy"
    stream_view: Literal["rgb", "delta", "auto"] = "auto"
    predict_horizon_ms: int = Field(default=100, ge=50, le=250)


class BenchmarkStartRequest(BaseModel):
    """Benchmark session start request."""

    name: Optional[str] = None
    scenario: Optional[str] = None
    notes: Optional[str] = None


try:
    from src.simulation.drone_sim import DroneSimulation

    SIMULATION_AVAILABLE = True
except ImportError:
    SIMULATION_AVAILABLE = False

    class DroneSimulation:
        """Fallback type when simulation dependencies are unavailable."""

        pass


app = FastAPI(title="AEROS API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


inference_engine: Optional[InferenceEngine] = None
passthrough_inference_engine = PassthroughInferenceEngine()
preprocessor: Optional[ImagePreprocessor] = None
pid_controller: Optional[PIDController] = None
simulation: Optional[DroneSimulation] = None
use_simulation: bool = False
physics_loop_task: Optional[asyncio.Task] = None
authoritative_stream: Optional[AuthoritativeStreamRuntime] = None
pid_gains = PIDGains(kp=1.0, ki=0.1, kd=0.5)
runtime_config = RuntimeConfig()

camera_index: int = 0
benchmark_recorder = BenchmarkRecorder(project_root / "benchmarks" / "runs")


def get_inference_engine():
    return inference_engine or passthrough_inference_engine


def get_preprocessor() -> Optional[ImagePreprocessor]:
    return preprocessor


def get_pid_controller() -> Optional[PIDController]:
    return pid_controller


def get_simulation() -> Optional[DroneSimulation]:
    return simulation


def get_use_simulation() -> bool:
    return use_simulation


def get_runtime_mode() -> str:
    return runtime_config.runtime_mode


def get_stream_view() -> str:
    return runtime_config.stream_view


def get_predict_horizon_ms() -> int:
    return runtime_config.predict_horizon_ms

def get_pid_gains() -> PIDGains:
    return pid_gains

def get_camera_index() -> int:
    return camera_index


def should_keep_runtime_running() -> bool:
    return True


def reset_runtime_state() -> None:
    """Reset stateful pipeline components after source or mode changes."""
    if preprocessor is not None:
        preprocessor.reset()

    if pid_controller is not None:
        pid_controller.reset()


@app.on_event("startup")
async def startup():
    """Initialize shared application services."""
    global authoritative_stream, pid_controller, preprocessor

    preprocessor = ImagePreprocessor()
    pid_controller = PIDController(kp=pid_gains.kp, ki=pid_gains.ki, kd=pid_gains.kd)
    authoritative_stream = AuthoritativeStreamRuntime(
        get_inference_engine=get_inference_engine,
        get_preprocessor=get_preprocessor,
        get_pid_controller=get_pid_controller,
        get_simulation=get_simulation,
        get_use_simulation=get_use_simulation,
        get_runtime_mode=get_runtime_mode,
        get_stream_view=get_stream_view,
        get_predict_horizon_ms=get_predict_horizon_ms,
        get_camera_index=get_camera_index,
        should_keep_running=should_keep_runtime_running,
        ensure_physics_loop=ensure_physics_loop_running,
        benchmark_recorder=benchmark_recorder,
    )
    startup_error = await authoritative_stream.ensure_running()
    if startup_error is not None:
        print(f"WARNING: Authoritative stream failed to start on boot: {startup_error}")
    elif inference_engine is None:
        print("Authoritative stream is running in passthrough inference mode.")

    print("AEROS API started")


@app.on_event("shutdown")
async def shutdown():
    """Cleanup on shutdown."""
    global simulation

    if authoritative_stream is not None:
        await authoritative_stream.stop()

    await stop_physics_loop()

    if simulation is not None:
        simulation.close()
        simulation = None


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


@app.get("/runtime_config")
async def get_runtime_config():
    """Return the current authoritative runtime configuration."""
    return {"status": "success", "config": runtime_config.model_dump()}


@app.post("/runtime_config")
async def set_runtime_config(config: RuntimeConfig):
    """Update runtime mode and visualization behavior."""
    global runtime_config

    if config.runtime_mode == "omega":
        return JSONResponse(
            status_code=400,
            content={
                "status": "error",
                "message": (
                    "runtime_mode=omega is reserved for the Phase 2 predictor. "
                    "Use omega_shadow for temporal encoding in the current build."
                ),
            },
        )

    runtime_config = config
    reset_runtime_state()
    benchmark_recorder.record_event(
        "runtime_config_updated",
        {"config": runtime_config.model_dump()},
    )

    if authoritative_stream is not None:
        await authoritative_stream.restart()

    return {
        "status": "success",
        "message": "Runtime configuration updated",
        "config": runtime_config.model_dump(),
    }


@app.post("/load_model")
async def load_model(model_path: str, use_onnx: bool = False):
    """Load an inference model from the models directory."""
    global inference_engine

    model_path_obj = Path(model_path)
    if model_path_obj.is_absolute():
        return JSONResponse(
            status_code=400,
            content={"status": "error", "message": "Model path must be relative"},
        )

    full_path = (project_root / "models" / model_path_obj).resolve()
    models_dir = (project_root / "models").resolve()
    if not str(full_path).startswith(str(models_dir)):
        return JSONResponse(
            status_code=403,
            content={
                "status": "error",
                "message": "Model path must be within models/ directory",
            },
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
        reset_runtime_state()
        benchmark_recorder.record_event(
            "model_loaded",
            {
                "model_path": str(full_path.relative_to(project_root)),
                "use_onnx": use_onnx,
            },
        )
        if authoritative_stream is not None:
            await authoritative_stream.restart()
        return {
            "status": "success",
            "message": f"Model loaded from {model_path}",
            "use_onnx": use_onnx,
        }
    except Exception as exc:
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": str(exc)},
        )


async def ensure_physics_loop_running() -> None:
    """Start the shared physics loop when simulation is active."""
    global physics_loop_task

    if not use_simulation or simulation is None:
        return

    if physics_loop_task is None or physics_loop_task.done():
        physics_loop_task = asyncio.create_task(
            physics_loop(),
            name="aeros-physics-loop",
        )


async def stop_physics_loop() -> None:
    """Stop the shared physics loop."""
    global physics_loop_task

    if physics_loop_task is None:
        return

    task = physics_loop_task
    physics_loop_task = None
    task.cancel()

    try:
        await task
    except asyncio.CancelledError:
        pass
    except Exception as exc:
        print(f"WARNING: Physics loop ended with error during cleanup: {exc}")
        print(traceback.format_exc())


async def physics_loop():
    """Run simulation stepping at a fixed cadence independent of clients."""
    global physics_loop_task

    current_task = asyncio.current_task()
    try:
        while use_simulation and simulation is not None:
            simulation.step()
            await asyncio.sleep(1.0 / 60.0)
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        print(f"Physics loop error: {exc}")
        print(traceback.format_exc())
    finally:
        if physics_loop_task is current_task:
            physics_loop_task = None


@app.post("/start_simulation")
async def start_simulation(
    gui: Optional[bool] = None,
    enable_obstacles: bool = True,
    enable_turns: bool = True,
    lighting_variation: float = 0.3,
):
    """Start the simulation source."""
    global simulation, use_simulation

    if not SIMULATION_AVAILABLE:
        return JSONResponse(
            status_code=503,
            content={
                "status": "error",
                "message": (
                    "Simulation is unavailable because PyBullet is not installed. "
                    "Native macOS installs currently skip it due to upstream "
                    "Bullet build failures on recent Apple SDKs. Use Docker/Linux "
                    "or a patched local build to enable simulation support."
                ),
            },
        )

    if gui is None:
        gui = os.getenv("DISPLAY") is not None and os.getenv("DISPLAY") != ""

    lighting_variation = max(0.0, min(1.0, lighting_variation))

    await stop_physics_loop()
    reset_runtime_state()
    if simulation is not None:
        simulation.close()
        simulation = None

    try:
        simulation = DroneSimulation(
            gui=gui,
            enable_obstacles=enable_obstacles,
            enable_turns=enable_turns,
            lighting_variation=lighting_variation,
        )
        use_simulation = True
        await ensure_physics_loop_running()
        benchmark_recorder.record_event(
            "simulation_started",
            {
                "gui": gui,
                "enable_obstacles": enable_obstacles,
                "enable_turns": enable_turns,
                "lighting_variation": lighting_variation,
            },
        )

        if authoritative_stream is not None:
            await authoritative_stream.restart()

        return {
            "status": "success",
            "message": (
                "Simulation started "
                f"(gui={gui}, obstacles={enable_obstacles}, lighting={lighting_variation})"
            ),
        }
    except Exception as exc:
        use_simulation = False
        if simulation is not None:
            simulation.close()
            simulation = None

        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": str(exc)},
        )


@app.post("/set_camera_source")
async def set_camera_source(index: int = 0):
    """Hot-swap the active camera source without stopping the heartbeat."""
    global camera_index

    if index < 0 or index > 9:
        return JSONResponse(
            status_code=400,
            content={
                "status": "error",
                "message": f"Camera index must be between 0 and 9, got {index}",
            },
        )

    previous_index = camera_index
    camera_index = index

    benchmark_recorder.record_event(
        "camera_source_changed",
        {"previous_index": previous_index, "new_index": index},
    )

    if authoritative_stream is not None:
        reset_runtime_state()
        restart_error = await authoritative_stream.restart()
        if restart_error is not None:
            return JSONResponse(
                status_code=500,
                content={"status": "error", "message": restart_error},
            )

    return {
        "status": "success",
        "message": f"Camera source switched from index {previous_index} to {index}",
        "camera_index": index,
    }


@app.post("/stop_simulation")
async def stop_simulation():
    """Stop the simulation source."""
    global simulation, use_simulation

    use_simulation = False
    await stop_physics_loop()
    reset_runtime_state()
    benchmark_recorder.record_event("simulation_stopped", {})

    if simulation is not None:
        simulation.close()
        simulation = None

    if authoritative_stream is not None:
        await authoritative_stream.restart()

    return {"status": "success", "message": "Simulation stopped"}


@app.post("/set_pid_gains")
async def set_pid_gains(gains: PIDGains):
    """Configure PID controller gains dynamically."""
    if pid_controller is None:
        return JSONResponse(
            status_code=400,
            content={"status": "error", "message": "PID controller not initialized"},
        )

    pid_controller.kp = gains.kp
    pid_controller.ki = gains.ki
    pid_controller.kd = gains.kd

    print(f"PID gains updated: Kp={gains.kp}, Ki={gains.ki}, Kd={gains.kd}")
    benchmark_recorder.record_event(
        "pid_gains_updated",
        {
            "kp": gains.kp,
            "ki": gains.ki,
            "kd": gains.kd,
        },
    )

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
    """Get current PID controller gains."""
    if pid_controller is None:
        return JSONResponse(
            status_code=400,
            content={"status": "error", "message": "PID controller not initialized"},
        )

    return {
        "status": "success",
        "gains": {
            "kp": pid_controller.kp,
            "ki": pid_controller.ki,
            "kd": pid_controller.kd,
        },
    }


def build_benchmark_metadata() -> dict:
    """Collect run metadata for benchmark exports."""
    model_metadata = {"mode": "passthrough"}
    if inference_engine is not None:
        model_metadata = {
            "mode": "checkpoint",
            "model_path": str(inference_engine.model_path.relative_to(project_root)),
            "use_onnx": inference_engine.use_onnx,
        }

    return {
        "runtime_config": runtime_config.model_dump(),
        "use_simulation": use_simulation,
        "model": model_metadata,
    }


@app.get("/benchmark/status")
async def get_benchmark_status():
    """Return the current benchmark session status."""
    return {"status": "success", "benchmark": benchmark_recorder.status()}


@app.get("/benchmark/latest")
async def get_latest_benchmark():
    """Return the most recent completed benchmark result."""
    latest = benchmark_recorder.latest_result()
    if latest is None:
        return JSONResponse(
            status_code=404,
            content={"status": "error", "message": "No benchmark result is available yet."},
        )

    return {"status": "success", "benchmark": latest}


@app.post("/benchmark/start")
async def start_benchmark(request: BenchmarkStartRequest):
    """Start a benchmark session and keep the runtime alive while it records."""
    if authoritative_stream is None:
        return JSONResponse(
            status_code=503,
            content={"status": "error", "message": "Authoritative runtime is unavailable."},
        )

    validation_error = authoritative_stream.validate_prerequisites()
    if validation_error is not None:
        return JSONResponse(
            status_code=400,
            content={"status": "error", "message": validation_error},
        )

    try:
        benchmark = benchmark_recorder.start_session(
            name=request.name,
            scenario=request.scenario,
            notes=request.notes,
            metadata=build_benchmark_metadata(),
        )
    except ValueError as exc:
        return JSONResponse(
            status_code=409,
            content={"status": "error", "message": str(exc)},
        )

    error = await authoritative_stream.ensure_running()
    if error is not None:
        benchmark_recorder.stop_session(export=False)
        await authoritative_stream.maybe_stop_when_idle()
        return JSONResponse(
            status_code=400,
            content={"status": "error", "message": error},
        )

    return {
        "status": "success",
        "message": "Benchmark session started",
        "benchmark": benchmark,
    }


@app.post("/benchmark/stop")
async def stop_benchmark(export: bool = True):
    """Stop the active benchmark session and optionally export it."""
    try:
        result = benchmark_recorder.stop_session(export=export)
    except ValueError as exc:
        return JSONResponse(
            status_code=400,
            content={"status": "error", "message": str(exc)},
        )

    if authoritative_stream is not None:
        await authoritative_stream.maybe_stop_when_idle()

    return {
        "status": "success",
        "message": "Benchmark session stopped",
        "benchmark": result,
    }


@app.websocket("/ws/test")
async def websocket_test(websocket: WebSocket):
    """Simple WebSocket test endpoint."""
    try:
        await websocket.accept()
        await websocket.send_json({"message": "WebSocket test successful"})
    finally:
        await websocket.close()


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """Passive WebSocket subscriber for authoritative stream snapshots."""
    global authoritative_stream

    print(f"Incoming WebSocket connection attempt from {websocket.client}")
    try:
        await websocket.accept()
    except Exception as exc:
        print(f"ERROR: Failed to accept WebSocket connection: {exc}")
        return

    if authoritative_stream is None:
        await websocket.send_json(
            {"type": "error", "error": "Authoritative stream runtime is unavailable."}
        )
        await websocket.close()
        return

    await authoritative_stream.register_client()
    last_seq_id = 0

    try:
        error = await authoritative_stream.ensure_running()
        if error is not None:
            await websocket.send_json({"type": "error", "error": error})
            return

        last_seq_id = authoritative_stream.current_seq_id()
        print("WebSocket connection established - streaming shared snapshots")

        while True:
            snapshot = await authoritative_stream.wait_for_snapshot(last_seq_id)
            last_seq_id = snapshot.seq_id

            if snapshot.error is not None:
                await websocket.send_json({"type": "error", "error": snapshot.error})
                break

            if snapshot.telemetry is not None:
                await websocket.send_json(snapshot.telemetry)

            if snapshot.frame_bytes is not None:
                await websocket.send_bytes(snapshot.frame_bytes)

    except WebSocketDisconnect:
        print("WebSocket disconnected (client closed)")
    except Exception as exc:
        print(f"CRITICAL ERROR: WebSocket stream error: {exc}")
        print(traceback.format_exc())
        try:
            await websocket.send_json({"type": "error", "error": str(exc)})
        except Exception:
            pass
    finally:
        await authoritative_stream.unregister_client()
        try:
            await websocket.close()
        except Exception:
            pass
