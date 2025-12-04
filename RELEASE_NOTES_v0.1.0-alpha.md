# AEROS v0.1.0-alpha - Pre-Release

**Release Date**: November 24, 2024

##  First Pre-Release

This is the initial pre-release of AEROS (Autonomy Pipeline for Lightweight Drone Navigation). AEROS is a complete visual autonomy stack designed to process real-time camera input, estimate heading, and execute stable navigation commands.

##  Key Features

### Core Components
- **CNN-based Heading Estimation**: Lightweight MobileNet-style architecture (~500K-1M parameters)
- **Real-time Processing**: WebSocket streaming at 30 FPS with <100ms latency
- **PyBullet Simulation**: Full 3D corridor environment with physics-based drone control
- **React Dashboard**: Modern web interface with live camera feed and telemetry
- **Docker Support**: Multi-container setup with health checks and auto-detection

### Control & Navigation
- **PID Controller**: Configurable heading correction with anti-windup
- **Torque-based Physics**: Realistic drone dynamics (mass, inertia, drag)
- **Synthetic Data Generation**: Automated training dataset creation

### Developer Experience
- **Comprehensive Documentation**: Architecture diagrams, API docs, quick start guide
- **Makefile Automation**: One-command setup, training, and deployment
- **Unit Tests**: Test coverage for core components
- **ONNX Export**: Edge device deployment support

##  Recent Critical Fixes

This release includes fixes for several critical issues:

1. **Docker Compatibility**: Auto-detects headless mode, prevents crashes in containers
2. **Physics Stability**: Fixed time warp bug where simulation speed changed with client count
3. **Color Accuracy**: Fixed RGB/BGR mismatch that caused model accuracy issues
4. **PID Control**: Eliminated startup derivative spike
5. **Security**: Added path validation to prevent traversal attacks
6. **Performance**: Optimized preprocessing (50-200ms → <5ms per frame)
7. **Memory Management**: Improved React video handling for high-frequency streams

##  Installation

```bash
# Clone repository
git clone https://github.com/Forgingalex/aeros.git
cd aeros

# Checkout release
git checkout v0.1.0-alpha

# Install dependencies
make install

# Generate training data
make generate-data

# Train model
make train

# Start API server
make run-api

# Start dashboard (in another terminal)
make run-web
```

Or use Docker:
```bash
docker-compose up -d
```

##  Performance Targets

- **Inference**: ≥15 FPS on CPU, ≥30 FPS on GPU
- **Latency**: <100ms end-to-end
- **Model Size**: <10MB (ONNX format)
- **Accuracy**: MAE <0.1 radians on validation set

##  Known Limitations

This is a **pre-release** version. Please note:

- Model trained on synthetic data only (not real-world images)
- Simulation physics simplified (not full aerodynamics)
- Not yet tested on physical drones
- Webcam access in Docker requires privileged mode
- PID controller tuned for simulation (may need retuning for hardware)

** Do not use in production or on real drones without extensive testing.**

##  What's Next

Planned for future releases:
- Real-world dataset collection and training
- Enhanced simulation physics (aerodynamics, wind, turbulence)
- ROS2 integration
- Edge device deployment guides
- Model quantization for embedded systems
- Multi-drone simulation support

##  Documentation

- [README.md](README.md) - Full documentation and quick start
- [CHANGELOG.md](CHANGELOG.md) - Detailed change history
- [API Documentation](http://localhost:8000/docs) - Interactive API docs (when server running)

##  Contributing

We welcome contributions! Please:
1. Fork the repository
2. Create a feature branch
3. Submit a pull request

##  License

This project is licensed under the MIT License - see [LICENSE](LICENSE) file for details.

##  Acknowledgments

Built with:
- PyTorch for deep learning
- FastAPI for backend
- React for frontend
- PyBullet for simulation
- OpenCV for image processing

---

**Download**: [v0.1.0-alpha](https://github.com/Forgingalex/aeros/releases/tag/v0.1.0-alpha)

**Report Issues**: [GitHub Issues](https://github.com/Forgingalex/aeros/issues)





