# Changelog

All notable changes to the AEROS project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0-alpha] - 2024-11-24

### Added
- Initial release of AEROS autonomy pipeline
- CNN-based heading estimation model (MobileNet-style architecture)
- Real-time WebSocket streaming for telemetry and video frames
- PyBullet simulation environment with corridor navigation
- React dashboard with live camera feed and telemetry visualization
- Docker containerization with multi-container setup
- PID controller for heading correction with configurable gains
- Synthetic data generation pipeline for training
- ONNX export functionality for edge deployment
- FastAPI backend with REST and WebSocket endpoints
- Comprehensive documentation with architecture diagrams
- Makefile for build automation
- Unit tests for preprocessing and model components
- Health check endpoints for Docker orchestration

### Fixed
- **Docker Headless Mode**: Auto-detection of headless environment, prevents crashes in containers
- **Physics Time Warp**: Moved simulation stepping to background asyncio task, prevents speed changes with multiple clients
- **RGB/BGR Mismatch**: Fixed color channel conversion in inference pipeline (model now receives correct RGB input)
- **PID Startup Kick**: Initialize `_last_error` on first update to prevent derivative spike
- **Security**: Added path validation to `/load_model` endpoint to prevent path traversal attacks
- **Performance**: Replaced expensive `fastNlMeansDenoisingColored` with `GaussianBlur` for real-time performance
- **React Memory**: Replaced Blob URLs with `createImageBitmap` and canvas to reduce GC pressure
- **Simulation Physics**: Improved physics model using torque/force instead of direct velocity setting

### Changed
- Default simulation GUI mode changed to `False` (auto-detects Docker environment)
- Preprocessing denoising now uses Gaussian blur instead of NLM for better performance
- Model path validation restricts loading to `models/` directory only

### Known Limitations
- Simulation physics simplified (torque-based control, not full aerodynamics)
- Model trained on synthetic corridor data only (not real-world images)
- Webcam access in Docker requires privileged mode and device mapping
- ONNX Runtime and PyBullet are optional dependencies (graceful degradation)
- Not yet tested on physical drones
- Model accuracy may vary with different lighting/environments
- PID controller tuned for simulation, may need retuning for real hardware

### Technical Details
- **Model Size**: ~500K-1M parameters (lightweight for edge deployment)
- **Target FPS**: ≥15 FPS on CPU, ≥30 FPS on GPU
- **Target Latency**: <100ms end-to-end
- **Model Format**: PyTorch (.pth) and ONNX (.onnx) supported
- **Python Version**: 3.10+
- **Node.js Version**: 18+

---

## [Unreleased]

### Added
- **PID Control Panel**: Interactive dashboard component for real-time PID gain adjustment
  - Real-time sliders and number inputs for Kp, Ki, Kd gains
  - 5 preset configurations (Default, Aggressive, Smooth, Fast Response, Stable)
  - Current values display showing active PID gains
  - Reset button to restore default values
  - Apply button for manual gain updates
- **PID API Endpoints**: REST endpoints for PID controller configuration
  - `GET /get_pid_gains` - Retrieve current PID controller gains
  - `POST /set_pid_gains` - Update PID controller gains dynamically
- **Dashboard Redesign**: Premium aviation-grade interface
  - New AEROS design system with CSS variables
  - Dark mode aerospace aesthetic
  - Updated logo with geometric flight vector icon
  - Improved component styling and layout
- **Enhanced Error Handling**: Comprehensive error handling for WebSocket endpoint
  - Multi-layer try-except blocks for frame processing
  - Detailed traceback logging for all errors
  - Graceful error recovery for non-critical failures
  - Improved error messages and diagnostics

### Fixed
- **WebSocket Stability**: Improved error handling prevents connection drops
- **PID Gain Updates**: Real-time gain updates now work correctly during operation

### Changed
- PID controller gains can now be adjusted dynamically via dashboard (no code changes required)
- Dashboard UI redesigned with new AEROS brand identity

### Planned
- Real-world dataset collection and training
- Enhanced simulation physics (aerodynamics, wind, turbulence)
- ROS2 integration
- Edge device deployment guides
- Model quantization for embedded systems
- Multi-drone simulation support
- Advanced control algorithms (MPC, LQR)
- Real-time obstacle detection
- Path planning integration





