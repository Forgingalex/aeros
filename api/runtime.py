"""Shared authoritative stream runtime for camera/simulation streaming."""

from __future__ import annotations

import asyncio
import time
import traceback
from dataclasses import dataclass
from typing import Awaitable, Callable, Optional, TYPE_CHECKING

import cv2

if TYPE_CHECKING:
    from src.control.pid_controller import PIDController
    from src.preprocessing.preprocessor import ImagePreprocessor
    from src.simulation.drone_sim import DroneSimulation
    from src.utils.benchmark import BenchmarkRecorder
    from src.utils.inference import InferenceEngine


@dataclass
class StreamSnapshot:
    """Latest authoritative stream payload for all connected clients."""

    seq_id: int
    telemetry: Optional[dict] = None
    frame_bytes: Optional[bytes] = None
    error: Optional[str] = None


class AuthoritativeStreamRuntime:
    """Runs one shared perception-control loop and broadcasts snapshots."""

    def __init__(
        self,
        *,
        get_inference_engine: Callable[[], Optional["InferenceEngine"]],
        get_preprocessor: Callable[[], Optional["ImagePreprocessor"]],
        get_pid_controller: Callable[[], Optional["PIDController"]],
        get_simulation: Callable[[], Optional["DroneSimulation"]],
        get_use_simulation: Callable[[], bool],
        get_runtime_mode: Callable[[], str],
        get_stream_view: Callable[[], str],
        get_predict_horizon_ms: Callable[[], int],
        should_keep_running: Callable[[], bool],
        ensure_physics_loop: Callable[[], Awaitable[None]],
        get_camera_index: Callable[[], int] = lambda: 0,
        benchmark_recorder: Optional["BenchmarkRecorder"] = None,
        frame_interval_s: float = 1.0 / 30.0,
    ):
        self._get_inference_engine = get_inference_engine
        self._get_preprocessor = get_preprocessor
        self._get_pid_controller = get_pid_controller
        self._get_simulation = get_simulation
        self._get_use_simulation = get_use_simulation
        self._get_runtime_mode = get_runtime_mode
        self._get_stream_view = get_stream_view
        self._get_predict_horizon_ms = get_predict_horizon_ms
        self._should_keep_running = should_keep_running
        self._ensure_physics_loop = ensure_physics_loop
        self._get_camera_index = get_camera_index
        self._benchmark_recorder = benchmark_recorder
        self._frame_interval_s = frame_interval_s

        self._lifecycle_lock = asyncio.Lock()
        self._snapshot_condition = asyncio.Condition()
        self._task: Optional[asyncio.Task] = None
        self._latest_snapshot = StreamSnapshot(seq_id=0)
        self._seq_id = 0
        self._client_count = 0

    async def register_client(self) -> None:
        """Track a new passive stream subscriber."""
        async with self._lifecycle_lock:
            self._client_count += 1
            if self._benchmark_recorder is not None:
                self._benchmark_recorder.record_event(
                    "client_connected",
                    {"client_count": self._client_count},
                )

    async def unregister_client(self) -> None:
        """Remove a passive subscriber without affecting the heartbeat."""
        async with self._lifecycle_lock:
            if self._client_count > 0:
                self._client_count -= 1
                if self._benchmark_recorder is not None:
                    self._benchmark_recorder.record_event(
                        "client_disconnected",
                        {"client_count": self._client_count},
                    )

    async def ensure_running(self) -> Optional[str]:
        """Start the authoritative loop if prerequisites are satisfied."""
        validation_error = self._validate_prerequisites()
        if validation_error is not None:
            return validation_error

        async with self._lifecycle_lock:
            if self._task is not None and not self._task.done():
                return None

            self._task = asyncio.create_task(
                self._run(),
                name="aeros-authoritative-stream",
            )

        return None

    def validate_prerequisites(self) -> Optional[str]:
        """Return a human-readable readiness error if the runtime cannot start."""
        return self._validate_prerequisites()

    async def restart(self) -> Optional[str]:
        """Restart the shared runtime to pick up source changes."""
        should_restart = False

        async with self._lifecycle_lock:
            should_restart = self._client_count > 0 or self._should_keep_running()
            task_to_cancel = self._task
            self._task = None

            if self._benchmark_recorder is not None:
                self._benchmark_recorder.record_event(
                    "runtime_restart_requested",
                    {
                        "client_count": self._client_count,
                        "keepalive": self._should_keep_running(),
                    },
                )

        await self._cancel_task(task_to_cancel)

        if should_restart:
            error = await self.ensure_running()
            if error is not None:
                await self.publish_error(error)
            return error

        return None

    async def stop(self) -> None:
        """Stop the shared runtime regardless of subscriber count."""
        async with self._lifecycle_lock:
            task_to_cancel = self._task
            self._task = None

        await self._cancel_task(task_to_cancel)

    async def maybe_stop_when_idle(self) -> None:
        """Stop the runtime only if no client or benchmark keepalive needs it."""
        task_to_cancel: Optional[asyncio.Task] = None

        async with self._lifecycle_lock:
            if (
                self._client_count == 0
                and self._task is not None
                and not self._should_keep_running()
            ):
                task_to_cancel = self._task
                self._task = None

        await self._cancel_task(task_to_cancel)

    def current_seq_id(self) -> int:
        """Return the latest published sequence number."""
        return self._latest_snapshot.seq_id

    async def wait_for_snapshot(self, last_seq_id: int) -> StreamSnapshot:
        """Wait until a newer snapshot than `last_seq_id` is available."""
        async with self._snapshot_condition:
            await self._snapshot_condition.wait_for(
                lambda: self._latest_snapshot.seq_id > last_seq_id
            )
            return self._latest_snapshot

    async def publish_error(self, error: str) -> None:
        """Publish a fatal stream error to all subscribers."""
        if self._benchmark_recorder is not None:
            self._benchmark_recorder.record_event(
                "runtime_error",
                {"error": error},
            )
        snapshot = StreamSnapshot(seq_id=self._next_seq_id(), error=error)
        await self._publish_snapshot(snapshot)

    def _validate_prerequisites(self) -> Optional[str]:
        if self._get_preprocessor() is None:
            return "Preprocessor not initialized."
        if self._get_pid_controller() is None:
            return "PID controller not initialized."
        return None

    def _next_seq_id(self) -> int:
        self._seq_id += 1
        return self._seq_id

    async def _publish_snapshot(self, snapshot: StreamSnapshot) -> None:
        async with self._snapshot_condition:
            self._latest_snapshot = snapshot
            self._snapshot_condition.notify_all()

    async def _cancel_task(self, task: Optional[asyncio.Task]) -> None:
        if task is None:
            return

        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        except Exception as exc:  # pragma: no cover - runtime guard
            print(f"WARNING: Stream runtime task ended with error during cleanup: {exc}")
            print(traceback.format_exc())

    async def _run(self) -> None:
        frame_count = 0
        consecutive_camera_failures = 0
        source_kind = (
            "simulation"
            if self._get_use_simulation() and self._get_simulation() is not None
            else "camera"
        )
        video_capture = None
        preprocessor = self._get_preprocessor()

        try:
            if preprocessor is not None:
                preprocessor.reset()

            if self._benchmark_recorder is not None:
                self._benchmark_recorder.record_event(
                    "runtime_loop_started",
                    {"source_kind": source_kind},
                )

            if source_kind == "camera":
                video_capture = self._open_camera_capture()
                if video_capture is None:
                    await self.publish_error(
                        "Could not open camera. Check if camera is available "
                        "and not in use by another application."
                    )
                    return
            else:
                await self._ensure_physics_loop()

            while True:
                loop_start = time.time()

                inference_engine = self._get_inference_engine()
                preprocessor = self._get_preprocessor()
                pid_controller = self._get_pid_controller()
                simulation = self._get_simulation()
                use_simulation = self._get_use_simulation()
                runtime_mode = self._get_runtime_mode()
                stream_view = self._get_stream_view()
                predict_horizon_ms = self._get_predict_horizon_ms()

                if inference_engine is None or preprocessor is None or pid_controller is None:
                    await self.publish_error(
                        "Stream prerequisites became unavailable during runtime."
                    )
                    return

                if runtime_mode == "omega":
                    await self.publish_error(
                        "runtime_mode=omega is not available yet. Use omega_shadow for temporal encoding."
                    )
                    return

                if source_kind == "simulation" and (not use_simulation or simulation is None):
                    if self._benchmark_recorder is not None:
                        self._benchmark_recorder.record_event(
                            "source_switch",
                            {"from": "simulation", "to": "camera"},
                        )
                    print("Simulation source changed; runtime loop exiting for restart.")
                    return

                if source_kind == "camera" and use_simulation:
                    if self._benchmark_recorder is not None:
                        self._benchmark_recorder.record_event(
                            "source_switch",
                            {"from": "camera", "to": "simulation"},
                        )
                    print("Camera source changed; runtime loop exiting for restart.")
                    return

                frame = self._read_frame(
                    source_kind=source_kind,
                    video_capture=video_capture,
                    simulation=simulation,
                )
                if frame is None:
                    if source_kind == "camera":
                        consecutive_camera_failures += 1
                        if consecutive_camera_failures < 5:
                            await asyncio.sleep(self._frame_interval_s)
                            continue
                    await self.publish_error("Failed to acquire a frame from the active source.")
                    return
                consecutive_camera_failures = 0

                temporal_features = None
                temporal_latency_ms = 0.0
                temporal_required = runtime_mode != "legacy" or stream_view == "delta"
                if temporal_required:
                    temporal_start = time.time()
                    try:
                        temporal_features = preprocessor.encode_temporal(frame)
                    except Exception as exc:  # pragma: no cover - runtime guard
                        print(f"ERROR: Temporal encoding failed: {exc}")
                        print(traceback.format_exc())
                        await asyncio.sleep(self._frame_interval_s)
                        continue
                    temporal_latency_ms = (time.time() - temporal_start) * 1000.0

                try:
                    preprocess_start = time.time()
                    preprocessed_frame = preprocessor.preprocess(frame)
                    preprocess_latency_ms = (time.time() - preprocess_start) * 1000.0
                except Exception as exc:  # pragma: no cover - runtime guard
                    print(f"ERROR: Preprocessing failed: {exc}")
                    print(traceback.format_exc())
                    await asyncio.sleep(self._frame_interval_s)
                    continue

                inference_passthrough = bool(
                    getattr(inference_engine, "is_passthrough", False)
                )
                try:
                    inference_start = time.time()
                    heading = inference_engine.predict(preprocessed_frame)
                    inference_latency_ms = (time.time() - inference_start) * 1000.0
                except Exception as exc:  # pragma: no cover - runtime guard
                    print(f"ERROR: Inference failed: {exc}")
                    print(traceback.format_exc())
                    await asyncio.sleep(self._frame_interval_s)
                    continue

                try:
                    control_start = time.time()
                    control_output = pid_controller.compute_control(heading)
                    control_latency_ms = (time.time() - control_start) * 1000.0
                except Exception as exc:  # pragma: no cover - runtime guard
                    print(f"ERROR: PID control computation failed: {exc}")
                    print(traceback.format_exc())
                    control_output = 0.0
                    control_latency_ms = 0.0

                if inference_passthrough:
                    control_output = 0.0

                drone_state = None
                if use_simulation and simulation is not None:
                    try:
                        drone_state = simulation.get_drone_state()
                        simulation.apply_control(control_output)
                    except Exception as exc:  # pragma: no cover - runtime guard
                        print(f"WARNING: Simulation state/control failed: {exc}")
                        print(traceback.format_exc())

                try:
                    fps = inference_engine.get_fps()
                    latency = inference_engine.get_latency()
                except Exception as exc:  # pragma: no cover - runtime guard
                        print(f"WARNING: Failed to collect metrics: {exc}")
                        fps = 0.0
                        latency = 0.0

                frame_for_stream = frame
                frame_kind = "rgb"
                if temporal_features is not None and (
                    stream_view == "delta"
                    or (stream_view == "auto" and runtime_mode != "legacy")
                ):
                    frame_for_stream = temporal_features.delta_vis
                    frame_kind = "delta_vis"

                success, buffer = cv2.imencode(
                    ".jpg",
                    frame_for_stream,
                    [cv2.IMWRITE_JPEG_QUALITY, 80],
                )
                if not success:
                    await self.publish_error("Failed to encode stream frame.")
                    return

                shadow = None
                if runtime_mode == "omega_shadow" and temporal_features is not None:
                    shadow = {
                        "mode": "omega_shadow",
                        "warmup": temporal_features.warmup,
                        "motion_energy": temporal_features.motion_energy,
                        "temporal_latency_ms": temporal_latency_ms,
                    }

                loop_latency_ms = (time.time() - loop_start) * 1000.0
                seq_id = self._next_seq_id()
                authoritative_warmup = (
                    True
                    if inference_passthrough
                    else (
                        temporal_features.warmup
                        if temporal_features is not None
                        else False
                    )
                )
                benchmark_payload = {
                    "type": "telemetry",
                    "runtime_mode": runtime_mode,
                    "authoritative_path": "legacy",
                    "source_kind": source_kind,
                    "frame_kind": frame_kind,
                    "seq_id": seq_id,
                    "source_ts_ms": int(loop_start * 1000),
                    "warmup": authoritative_warmup,
                    "heading": float(heading),
                    "heading_rate": 0.0,
                    "control_output": float(0.0 if inference_passthrough else control_output),
                    "fps": float(fps),
                    "latency_ms": float(latency),
                    "preprocess_latency_ms": float(preprocess_latency_ms),
                    "inference_latency_ms": float(inference_latency_ms),
                    "motion_energy": float(
                        temporal_features.motion_energy if temporal_features is not None else 0.0
                    ),
                    "temporal_latency_ms": float(temporal_latency_ms),
                    "control_latency_ms": float(control_latency_ms),
                    "loop_latency_ms": float(loop_latency_ms),
                    "confidence": None,
                    "predict_horizon_ms": int(predict_horizon_ms),
                    "predicted_pose": None,
                    "safety_margin": None,
                    "tube_violation": None,
                    "frame_count": frame_count,
                    "drone_state": drone_state,
                    "shadow": shadow,
                }
                telemetry = self._build_canonical_telemetry(benchmark_payload)

                snapshot = StreamSnapshot(
                    seq_id=seq_id,
                    telemetry=telemetry,
                    frame_bytes=buffer.tobytes(),
                )
                await self._publish_snapshot(snapshot)
                if self._benchmark_recorder is not None:
                    self._benchmark_recorder.record_frame(benchmark_payload)
                frame_count += 1

                elapsed = time.time() - loop_start
                await asyncio.sleep(max(0.0, self._frame_interval_s - elapsed))

        except asyncio.CancelledError:
            raise
        except Exception as exc:  # pragma: no cover - runtime guard
            print(f"CRITICAL ERROR: Authoritative stream runtime failed: {exc}")
            print(traceback.format_exc())
            await self.publish_error(str(exc))
        finally:
            if video_capture is not None:
                video_capture.release()
            if preprocessor is not None:
                preprocessor.reset()
            if self._benchmark_recorder is not None:
                self._benchmark_recorder.record_event(
                    "runtime_loop_stopped",
                    {"source_kind": source_kind},
                )

    def _build_canonical_telemetry(self, metrics: dict) -> dict:
        """Project internal runtime metrics onto the canonical WebSocket schema."""
        return {
            "type": "telemetry",
            "runtime_mode": str(metrics["runtime_mode"]),
            "authoritative_path": str(metrics["authoritative_path"]),
            "frame_kind": str(metrics["frame_kind"]),
            "seq_id": int(metrics["seq_id"]),
            "source_ts_ms": int(metrics["source_ts_ms"]),
            "warmup": bool(metrics["warmup"]),
            "heading": float(metrics["heading"]),
            "heading_rate": float(metrics["heading_rate"]),
            "control_output": float(metrics["control_output"]),
            "fps": float(metrics["fps"]),
            "latency_ms": float(metrics["latency_ms"]),
            "motion_energy": float(metrics["motion_energy"]),
            "confidence": metrics["confidence"],
            "predict_horizon_ms": int(metrics["predict_horizon_ms"]),
            "predicted_pose": metrics["predicted_pose"],
            "safety_margin": metrics["safety_margin"],
            "tube_violation": metrics["tube_violation"],
            "drone_state": metrics["drone_state"],
            "shadow": metrics["shadow"],
        }

    def _open_camera_capture(self):
        """Open the default camera with macOS-friendly backend fallbacks."""
        backend_candidates = []
        if hasattr(cv2, "CAP_AVFOUNDATION"):
            backend_candidates.append(("avfoundation", cv2.CAP_AVFOUNDATION))
        if hasattr(cv2, "CAP_ANY"):
            backend_candidates.append(("any", cv2.CAP_ANY))
        backend_candidates.append(("default", None))

        camera_idx = self._get_camera_index()
        attempted_backends = []
        for backend_name, backend in backend_candidates:
            attempted_backends.append(backend_name)
            try:
                if backend is None:
                    capture = cv2.VideoCapture(camera_idx)
                else:
                    capture = cv2.VideoCapture(camera_idx, backend)
            except Exception as exc:  # pragma: no cover - runtime guard
                print(f"WARNING: Failed to initialize camera backend {backend_name}: {exc}")
                continue

            if capture is not None and capture.isOpened():
                capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                print(f"Camera opened successfully using backend: {backend_name}")
                return capture

            if capture is not None:
                capture.release()

        print(
            "WARNING: Unable to open camera with backends: "
            + ", ".join(attempted_backends)
        )
        return None

    def _read_frame(
        self,
        *,
        source_kind: str,
        video_capture,
        simulation: Optional["DroneSimulation"],
    ):
        if source_kind == "simulation":
            try:
                frame = simulation.get_camera_image()
                return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            except Exception as exc:  # pragma: no cover - runtime guard
                print(f"ERROR: Failed to get simulation frame: {exc}")
                print(traceback.format_exc())
                return None

        try:
            success, frame = video_capture.read()
        except Exception as exc:  # pragma: no cover - runtime guard
            print(f"ERROR: Camera read failed: {exc}")
            print(traceback.format_exc())
            return None

        if not success:
            print("WARNING: Failed to read frame from camera")
            return None

        return frame
