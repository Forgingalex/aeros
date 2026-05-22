"""Tests for authoritative runtime contract helpers."""

from src.control.pid_controller import PIDController
from src.preprocessing.preprocessor import ImagePreprocessor

from api.runtime import AuthoritativeStreamRuntime


def make_runtime() -> AuthoritativeStreamRuntime:
    """Create a runtime with lightweight test doubles."""
    return AuthoritativeStreamRuntime(
        get_inference_engine=lambda: None,
        get_preprocessor=lambda: ImagePreprocessor(),
        get_pid_controller=lambda: PIDController(),
        get_simulation=lambda: None,
        get_use_simulation=lambda: False,
        get_runtime_mode=lambda: "legacy",
        get_stream_view=lambda: "rgb",
        get_predict_horizon_ms=lambda: 100,
        should_keep_running=lambda: True,
        ensure_physics_loop=lambda: None,
    )


def test_runtime_validation_allows_passthrough_without_model():
    """Phase 0 turnkey mode should not require a loaded checkpoint."""
    runtime = make_runtime()

    assert runtime.validate_prerequisites() is None


def test_runtime_builds_canonical_websocket_schema():
    """WebSocket telemetry should match the canonical handbook contract."""
    runtime = make_runtime()
    payload = runtime._build_canonical_telemetry(
        {
            "type": "telemetry",
            "runtime_mode": "legacy",
            "authoritative_path": "legacy",
            "frame_kind": "rgb",
            "seq_id": 42,
            "source_ts_ms": 123456789,
            "warmup": True,
            "heading": 0.0,
            "heading_rate": 0.0,
            "control_output": 0.0,
            "fps": 0.0,
            "latency_ms": 0.0,
            "motion_energy": 0.0,
            "confidence": None,
            "predict_horizon_ms": 100,
            "predicted_pose": None,
            "safety_margin": None,
            "tube_violation": None,
            "drone_state": None,
            "shadow": None,
        }
    )

    assert list(payload.keys()) == [
        "type",
        "runtime_mode",
        "authoritative_path",
        "frame_kind",
        "seq_id",
        "source_ts_ms",
        "warmup",
        "heading",
        "heading_rate",
        "control_output",
        "fps",
        "latency_ms",
        "motion_energy",
        "confidence",
        "predict_horizon_ms",
        "predicted_pose",
        "safety_margin",
        "tube_violation",
        "drone_state",
        "shadow",
    ]
