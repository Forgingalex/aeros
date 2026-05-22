# Changelog

All notable changes to the AEROS project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0-omega-baseline] - 2026-05-22

This release establishes the baseline for the Omega transition. It abandons the foundational per-client alpha architecture in favor of a professional-grade, mission-ready perception stack.

### Added
- **Authoritative Heartbeat Loop**: A single `AuthoritativeStreamRuntime` background loop now owns the active video source. It executes frame acquisition, inference, control computation, and telemetry generation independently of WebSocket client connections.
- **Camera Hot-Swapping**: New `POST /set_camera_source?index=n` endpoint allows dynamic swapping of camera inputs (e.g., iPhone Continuity Camera) without killing the heartbeat or severing the WebSocket stream.
- **Aviation-Window HUD**: Completely rebuilt React dashboard utilizing zero-state animation constraints. Tapes slide physically behind stationary glass readouts to mimic jet avionics.
- **Kinetic Bore-Sight**: The center crosshair actively tracks drift. A Velocity Vector (Ghost Icon) shifts laterally based on heading rate telemetry, and AOA brackets pulse/scale dynamically based on motion energy.
- **PID API Endpoints**: REST endpoints for PID controller configuration (`GET /get_pid_gains`, `POST /set_pid_gains`).
- **PID Control Panel**: Interactive dashboard component for real-time PID gain adjustment with 5 preset configurations.

### Changed
- **Frontend Performance Restrictions**: Strict enforcement of the 5ms latency law. High-frequency HUD elements bypass React state entirely, updating via direct DOM mutations inside `requestAnimationFrame`.
- **Identity Update**: Rebranded strictly to AEROS. Applied holographic cyan `drop-shadow` styling and removed legacy text.

### Fixed
- **WebSocket Stability**: Decoupling the heartbeat from the WebSocket connection eliminates connection drop cascades.
- **Memory Pressure**: Active rejection of `URL.createObjectURL(Blob)` in favor of `createImageBitmap` drastically reduces frontend garbage collection spikes.

## [0.1.0-alpha] - 2024-11-24

My foundational base. This release proved that a lightweight CNN can maintain drone stability when latency is treated as the primary constraint.

### Added
- Initial release of the AEROS autonomy pipeline.
- CNN-based heading estimation model (MobileNet-style architecture).
- Real-time WebSocket streaming for telemetry and video frames.
- PyBullet simulation environment with corridor navigation.
- Docker containerization with multi-container setup.
- PID controller for heading correction with configurable gains.
- ONNX export functionality for edge deployment.

### Fixed
- **Physics Time Warp**: Moved simulation stepping to a background asyncio task.
- **RGB/BGR Mismatch**: Fixed color channel conversion in inference pipeline.
- **PID Startup Kick**: Initialized `_last_error` on first update to prevent derivative spikes.

### Known Limitations
- Simulation physics are simplified (torque-based control, no full aerodynamics).
- macOS native PyBullet installation fails against recent Apple SDKs.
- Model relies entirely on synthetic corridor data.
- PID controller requires severe retuning for physical hardware.
