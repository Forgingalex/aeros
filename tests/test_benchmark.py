"""Tests for runtime benchmark recording."""

import json

from src.utils.benchmark import BenchmarkRecorder


def make_telemetry(seq_id: int, *, source_ts_ms: int, control_output: float = 0.0) -> dict:
    """Build a minimal telemetry payload for benchmark tests."""
    return {
        "seq_id": seq_id,
        "source_ts_ms": source_ts_ms,
        "runtime_mode": "legacy",
        "source_kind": "camera",
        "frame_kind": "rgb",
        "warmup": False,
        "heading": 0.1 * seq_id,
        "heading_rate": 0.0,
        "control_output": control_output,
        "fps": 20.0,
        "latency_ms": 8.0,
        "preprocess_latency_ms": 1.0,
        "inference_latency_ms": 2.0,
        "temporal_latency_ms": 0.0,
        "control_latency_ms": 0.5,
        "loop_latency_ms": 5.0,
        "motion_energy": 0.02 * seq_id,
    }


def test_benchmark_recorder_tracks_frame_gaps_and_exports(tmp_path):
    """Frame gaps should be detected and exported in the final summary."""
    recorder = BenchmarkRecorder(tmp_path)
    recorder.start_session(name="gap-test", scenario="frame-drop")
    recorder.record_frame(make_telemetry(1, source_ts_ms=1000, control_output=0.2))
    recorder.record_frame(make_telemetry(3, source_ts_ms=1050, control_output=-0.2))

    result = recorder.stop_session(export=True)

    assert result["summary"]["frames_recorded"] == 2
    assert result["summary"]["total_dropped_frames"] == 1
    assert result["summary"]["control_oscillation"]["sign_changes"] == 1
    assert result["export_path"] is not None

    exported = json.loads((tmp_path / f'{result["session"]["session_id"]}.json').read_text())
    assert exported["summary"]["total_dropped_frames"] == 1
    assert any(event["event_type"] == "frame_gap" for event in exported["events"])


def test_benchmark_status_and_latest_result(tmp_path):
    """Status should reflect active sessions and latest_result should persist after stop."""
    recorder = BenchmarkRecorder(tmp_path)
    status = recorder.start_session(name="status-test")

    assert status["active"] is True
    assert status["name"] == "status-test"

    recorder.record_event("manual_note", {"label": "hello"})
    recorder.record_frame(make_telemetry(10, source_ts_ms=2000))
    recorder.stop_session(export=False)

    latest = recorder.latest_result()
    assert latest is not None
    assert latest["session"]["name"] == "status-test"
    assert latest["summary"]["frames_recorded"] == 1
