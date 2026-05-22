# AEROS v0.1.0-alpha - The Foundational Base

**Release Date**: November 24, 2024

This release serves as my **foundational base**. It proved that a lightweight, camera-only perception stack can maintain drone stability when latency is treated as the primary constraint. 

While AEROS has since advanced into the Omega architecture (which rebuilds the engine for professional-grade autonomy and decoupling), this alpha release established the 5ms stage budget laws that I still abide by today.

## Key Features

### Core Components
- **CNN-based Heading Estimation**: Lightweight MobileNet-style architecture (~500K-1M parameters).
- **Real-time Processing**: WebSocket streaming at 30 FPS with <100ms latency.
- **PyBullet Simulation**: Full 3D corridor environment with physics-based drone control.
- **Docker Support**: Multi-container setup with health checks and auto-detection.

### Control & Navigation
- **PID Controller**: Configurable heading correction with anti-windup.
- **Torque-based Physics**: Realistic drone dynamics (mass, inertia, drag).
- **Synthetic Data Generation**: Automated training dataset creation.

## Historical Critical Fixes

These fixes laid the groundwork for our current authoritative heartbeat:

1. **Physics Stability**: Fixed a time warp bug where simulation speed changed with client count. This was our first step toward decoupling.
2. **Color Accuracy**: Fixed RGB/BGR mismatch that caused model accuracy issues.
3. **PID Control**: Eliminated startup derivative spikes.
4. **Performance**: Optimized preprocessing (50-200ms → <5ms per frame).

## Known Limitations

I am honest about my flaws. Please note the following constraints of this foundational release:

- **Simulation on macOS**: PyBullet fails to build against recent Apple SDKs. Native macOS simulation is disabled. Use Docker/Linux for headless simulation.
- **Training Data**: The model is trained entirely on synthetic data. Real-world validation remains deferred to future edge deployments.
- **Physics**: Simulation physics are simplified (torque-based control, not full aerodynamics).
- **Control Tuning**: The PID controller is tuned for the simulator. Physical hardware requires severe retuning.

---

**Built to treat latency as a law.**
