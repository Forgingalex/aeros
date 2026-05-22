"""Runtime benchmark recording and export utilities."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from threading import Lock
import time
from typing import Any, Optional
import uuid

import numpy as np


@dataclass
class BenchmarkEvent:
    """A timestamped benchmark event."""

    ts_ms: int
    event_type: str
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class BenchmarkFrameRecord:
    """Per-frame metrics captured during a benchmark session."""

    seq_id: int
    source_ts_ms: int
    runtime_mode: str
    source_kind: str
    frame_kind: str
    warmup: bool
    heading: float
    heading_rate: float
    control_output: float
    fps: float
    latency_ms: float
    preprocess_latency_ms: float
    inference_latency_ms: float
    temporal_latency_ms: float
    control_latency_ms: float
    loop_latency_ms: float
    motion_energy: float
    dropped_frames: int


class BenchmarkRecorder:
    """Capture, summarize, and export runtime benchmark sessions."""

    def __init__(self, export_root: Path):
        self._export_root = Path(export_root)
        self._lock = Lock()
        self._active = False
        self._session_id: Optional[str] = None
        self._name: Optional[str] = None
        self._scenario: Optional[str] = None
        self._notes: Optional[str] = None
        self._metadata: dict[str, Any] = {}
        self._started_at_ms: Optional[int] = None
        self._stopped_at_ms: Optional[int] = None
        self._frames: list[BenchmarkFrameRecord] = []
        self._events: list[BenchmarkEvent] = []
        self._last_seq_id: Optional[int] = None
        self._latest_result: Optional[dict[str, Any]] = None
        self._latest_export_path: Optional[Path] = None

    def is_active(self) -> bool:
        """Return whether a benchmark session is currently recording."""
        with self._lock:
            return self._active

    def start_session(
        self,
        *,
        name: Optional[str] = None,
        scenario: Optional[str] = None,
        notes: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        """Start a fresh benchmark recording session."""
        with self._lock:
            if self._active:
                raise ValueError("A benchmark session is already active.")

            started_at_ms = self._now_ms()
            session_id = f"{started_at_ms}-{uuid.uuid4().hex[:8]}"
            session_name = name or f"benchmark-{started_at_ms}"

            self._active = True
            self._session_id = session_id
            self._name = session_name
            self._scenario = scenario
            self._notes = notes
            self._metadata = metadata.copy() if metadata is not None else {}
            self._started_at_ms = started_at_ms
            self._stopped_at_ms = None
            self._frames = []
            self._events = []
            self._last_seq_id = None

            self._record_event_unlocked(
                "benchmark_start",
                {
                    "name": session_name,
                    "scenario": scenario,
                },
            )

            return self._status_unlocked()

    def stop_session(self, *, export: bool = True) -> dict[str, Any]:
        """Stop the active session and optionally export it to disk."""
        with self._lock:
            if not self._active:
                raise ValueError("No benchmark session is currently active.")

            self._stopped_at_ms = self._now_ms()
            self._record_event_unlocked("benchmark_stop", {})
            self._active = False

            result = self._build_result_unlocked()
            export_path = None
            if export:
                self._export_root.mkdir(parents=True, exist_ok=True)
                export_path = self._export_root / f"{self._session_id}.json"
                export_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
                self._latest_export_path = export_path
                result["export_path"] = str(export_path)
            else:
                result["export_path"] = None

            self._latest_result = result
            return result

    def record_event(self, event_type: str, details: Optional[dict[str, Any]] = None) -> None:
        """Append a benchmark event if a session is active."""
        with self._lock:
            if not self._active:
                return
            self._record_event_unlocked(event_type, details or {})

    def record_frame(self, telemetry: dict[str, Any]) -> None:
        """Capture one telemetry frame if a session is active."""
        with self._lock:
            if not self._active:
                return

            seq_id = int(telemetry.get("seq_id", 0))
            dropped_frames = 0
            if self._last_seq_id is not None:
                if seq_id > self._last_seq_id + 1:
                    dropped_frames = seq_id - self._last_seq_id - 1
                    self._record_event_unlocked(
                        "frame_gap",
                        {
                            "expected_seq_id": self._last_seq_id + 1,
                            "actual_seq_id": seq_id,
                            "dropped_frames": dropped_frames,
                        },
                    )
                elif seq_id <= self._last_seq_id:
                    self._record_event_unlocked(
                        "seq_reset",
                        {
                            "previous_seq_id": self._last_seq_id,
                            "new_seq_id": seq_id,
                        },
                    )

            record = BenchmarkFrameRecord(
                seq_id=seq_id,
                source_ts_ms=int(telemetry.get("source_ts_ms", self._now_ms())),
                runtime_mode=str(telemetry.get("runtime_mode", "legacy")),
                source_kind=str(telemetry.get("source_kind", "camera")),
                frame_kind=str(telemetry.get("frame_kind", "rgb")),
                warmup=bool(telemetry.get("warmup", False)),
                heading=float(telemetry.get("heading", 0.0)),
                heading_rate=float(telemetry.get("heading_rate", 0.0)),
                control_output=float(telemetry.get("control_output", 0.0)),
                fps=float(telemetry.get("fps", 0.0)),
                latency_ms=float(telemetry.get("latency_ms", 0.0)),
                preprocess_latency_ms=float(telemetry.get("preprocess_latency_ms", 0.0)),
                inference_latency_ms=float(telemetry.get("inference_latency_ms", 0.0)),
                temporal_latency_ms=float(telemetry.get("temporal_latency_ms", 0.0)),
                control_latency_ms=float(telemetry.get("control_latency_ms", 0.0)),
                loop_latency_ms=float(telemetry.get("loop_latency_ms", 0.0)),
                motion_energy=float(telemetry.get("motion_energy", 0.0)),
                dropped_frames=dropped_frames,
            )
            self._frames.append(record)
            self._last_seq_id = seq_id

    def status(self) -> dict[str, Any]:
        """Return the current session status."""
        with self._lock:
            return self._status_unlocked()

    def latest_result(self) -> Optional[dict[str, Any]]:
        """Return the most recent completed benchmark result."""
        with self._lock:
            return self._latest_result

    def latest_export_path(self) -> Optional[str]:
        """Return the most recent export path as a string."""
        with self._lock:
            if self._latest_export_path is None:
                return None
            return str(self._latest_export_path)

    def _build_result_unlocked(self) -> dict[str, Any]:
        summary = self._build_summary_unlocked()
        return {
            "session": {
                "session_id": self._session_id,
                "name": self._name,
                "scenario": self._scenario,
                "notes": self._notes,
                "metadata": self._metadata,
                "started_at_ms": self._started_at_ms,
                "stopped_at_ms": self._stopped_at_ms,
            },
            "summary": summary,
            "events": [asdict(event) for event in self._events],
            "frames": [asdict(frame) for frame in self._frames],
        }

    def _build_summary_unlocked(self) -> dict[str, Any]:
        end_ts_ms = self._stopped_at_ms or self._now_ms()
        started_at_ms = self._started_at_ms or end_ts_ms
        duration_s = max(0.0, (end_ts_ms - started_at_ms) / 1000.0)

        headings = np.array([frame.heading for frame in self._frames], dtype=np.float64)
        controls = np.array(
            [frame.control_output for frame in self._frames],
            dtype=np.float64,
        )

        total_dropped_frames = int(sum(frame.dropped_frames for frame in self._frames))

        source_ts = np.array(
            [frame.source_ts_ms for frame in self._frames],
            dtype=np.float64,
        )
        delivered_fps = 0.0
        if len(source_ts) >= 2:
            deltas_s = np.diff(source_ts) / 1000.0
            positive_deltas_s = deltas_s[deltas_s > 0]
            if positive_deltas_s.size > 0:
                delivered_fps = float(1.0 / np.median(positive_deltas_s))
        elif duration_s > 0 and self._frames:
            delivered_fps = float(len(self._frames) / duration_s)

        nonzero_controls = controls[np.abs(controls) > 1e-6]
        control_sign_changes = 0
        if nonzero_controls.size >= 2:
            control_sign_changes = int(
                np.sum(np.sign(nonzero_controls[1:]) != np.sign(nonzero_controls[:-1]))
            )

        return {
            "duration_s": duration_s,
            "frames_recorded": len(self._frames),
            "events_recorded": len(self._events),
            "total_dropped_frames": total_dropped_frames,
            "runtime_modes_seen": sorted({frame.runtime_mode for frame in self._frames}),
            "source_kinds_seen": sorted({frame.source_kind for frame in self._frames}),
            "frame_kinds_seen": sorted({frame.frame_kind for frame in self._frames}),
            "warmup_frames": int(sum(1 for frame in self._frames if frame.warmup)),
            "reported_fps_hz": self._series_stats([frame.fps for frame in self._frames]),
            "delivered_fps_hz": delivered_fps,
            "latency_ms": self._series_stats([frame.latency_ms for frame in self._frames]),
            "preprocess_latency_ms": self._series_stats(
                [frame.preprocess_latency_ms for frame in self._frames]
            ),
            "inference_latency_ms": self._series_stats(
                [frame.inference_latency_ms for frame in self._frames]
            ),
            "temporal_latency_ms": self._series_stats(
                [frame.temporal_latency_ms for frame in self._frames]
            ),
            "control_latency_ms": self._series_stats(
                [frame.control_latency_ms for frame in self._frames]
            ),
            "loop_latency_ms": self._series_stats(
                [frame.loop_latency_ms for frame in self._frames]
            ),
            "motion_energy": self._series_stats(
                [frame.motion_energy for frame in self._frames]
            ),
            "heading_smoothness_rad": {
                "mean_abs_delta": self._mean_abs_delta(headings),
                "p95_abs_delta": self._p95_abs_delta(headings),
            },
            "control_oscillation": {
                "mean_abs_delta": self._mean_abs_delta(controls),
                "p95_abs_delta": self._p95_abs_delta(controls),
                "sign_changes": control_sign_changes,
            },
            "unsupported_metrics": {
                "heading_error_rad": None,
                "ghost_projection_error_100ms": None,
                "safety_margin_stability": None,
            },
        }

    def _record_event_unlocked(self, event_type: str, details: dict[str, Any]) -> None:
        self._events.append(
            BenchmarkEvent(
                ts_ms=self._now_ms(),
                event_type=event_type,
                details=details,
            )
        )

    def _status_unlocked(self) -> dict[str, Any]:
        return {
            "active": self._active,
            "session_id": self._session_id,
            "name": self._name,
            "scenario": self._scenario,
            "notes": self._notes,
            "frames_recorded": len(self._frames),
            "events_recorded": len(self._events),
            "started_at_ms": self._started_at_ms,
            "latest_export_path": (
                str(self._latest_export_path) if self._latest_export_path is not None else None
            ),
        }

    @staticmethod
    def _series_stats(values: list[float]) -> dict[str, Optional[float]]:
        if not values:
            return {"median": None, "p95": None}
        array = np.array(values, dtype=np.float64)
        return {
            "median": float(np.median(array)),
            "p95": float(np.percentile(array, 95)),
        }

    @staticmethod
    def _mean_abs_delta(values: np.ndarray) -> float:
        if values.size < 2:
            return 0.0
        return float(np.mean(np.abs(np.diff(values))))

    @staticmethod
    def _p95_abs_delta(values: np.ndarray) -> float:
        if values.size < 2:
            return 0.0
        return float(np.percentile(np.abs(np.diff(values)), 95))

    @staticmethod
    def _now_ms() -> int:
        return int(time.time() * 1000)
