# AEROS Technical Documentation

This document defines the technical architecture of AEROS as it exists today. It details the transition from foundational alpha base to a professional-grade, mission-ready perception stack.

For the philosophical constraints and the future Omega roadmap, see [README.md](README.md).

---

## Architecture

### The Authoritative Heartbeat

The foundational alpha tied perception to WebSocket client connections. Today, AEROS runs on an **Authoritative Ownership Model**. 

A single `AuthoritativeStreamRuntime` background loop owns the active video source. It executes frame acquisition, inference, control computation, and telemetry generation at a fixed cadence. WebSocket subscribers are passive; they only observe the generated stream. This guarantees that whether there are ten clients connected or none, the physics simulation and control loop maintains strict 5ms deadline integrity. 

### Frontend HUD & Zero-State Performance

The React dashboard acts as a high-fidelity flight instrument, adhering to strict hardware efficiency guidelines:

- **Aviation-Window Pattern:** Digital readouts (HDG, FPS, Energy) are fixed in place. The physical tapes slide behind them, replicating real jet avionics.
- **Zero-State Animation:** I do not use React state for high-frequency DOM updates. Telemetry data feeds directly into a continuous `requestAnimationFrame` loop. 
- **Direct DOM Mutation:** Element transformations (`translateY` for tapes, `translateX` for the Velocity Vector, and `scaleY` for AOA brackets) bypass the Virtual DOM entirely.
- **Garbage Collection (GC) Mitigation:** I process the JPEG binary stream using `createImageBitmap` and draw it directly to an HTML5 `<canvas>`. I actively reject `URL.createObjectURL(Blob)` because it triggers destructive GC pauses during flight.

### Model Architecture

- **Input:** 224x224 RGB images
- **Architecture:** MobileNet-style depthwise separable convolutions
- **Layers:** 5 separable conv blocks (32→64→128→256→512 channels) + global average pooling + 2 FC layers
- **Output:** Single scalar (heading angle in radians)
- **Parameters:** ~500K-1M (lightweight for edge deployment)

### Control System

- PID controller with configurable Kp, Ki, Kd gains
- Real-time gain adjustment via REST API and HUD UI
- Torque-based physics simulation respecting inertia and mass
- Background physics loop running at fixed 60 Hz to prevent time warping

---

## What Is Implemented

I have established the following mission-ready capabilities:

- **Authoritative Heartbeat:** Single background loop decoupling logic from WebSocket I/O.
- **Camera Hot-Swapping:** `POST /set_camera_source` endpoint allows dynamic switching of the `cv2.VideoCapture` index without stopping the heartbeat.
- **High-Fidelity HUD:** Zero-state React implementation with aviation tapes, moving bore-sights, and holographic cyan visual styling.
- **Real-Time Preprocessing:** Gaussian blur and CLAHE, tuned to fit inside a <10ms budget.
- **PID Control:** Startup derivative spike protection and active torque-based physics control.

---

## API Endpoints

### REST Endpoints

- `GET /health` - Health check
- `POST /load_model` - Load inference model weights
- `POST /start_simulation?gui=true` - Start PyBullet simulation
- `POST /stop_simulation` - Stop simulation
- `POST /set_camera_source?index=n` - Hot-swap the active camera feed without dropping connection.
- `GET /get_pid_gains` - Get current PID controller gains
- `POST /set_pid_gains` - Update PID controller gains dynamically

### WebSocket Endpoints

- `WS /ws` - Real-time stream for telemetry and video frames. Sends JSON telemetry messages followed immediately by binary JPEG-encoded frames.

---

## Known Limitations & Technical Truths

I value honesty in my engineering constraints. Do not assume capabilities that are not explicitly documented.

- **Simulation on macOS:** PyBullet currently fails to build against recent Apple SDKs. Native macOS simulation is skipped. Use the Docker deployment for headless simulation.
- **Visual Drift Indicator:** The Velocity Vector (Ghost Icon) currently shifts laterally based on the raw heading rate telemetry. This is a visual UI effect and does not yet represent a true predictive motion physics calculation.
- **RGB Bias:** The current model evaluates static RGB frames. True temporal delta encoding is scheduled for the Omega phase.
- **Simulation Physics:** Simplified torque-based control only. No aerodynamics, wind, or turbulence modeling.
- **PID Tuning:** Gains are tuned for simulation. Real-world hardware will require severe retuning. 

---

## Setup & Deployment

### Local Execution

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cd web && npm install && cd ..
```

**Start the Heartbeat (Backend):**
```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

**Start the Instrument (Frontend):**
```bash
cd web && npm start
```

### Docker Deployment

For headless simulation or deployment on machines lacking local Python development environments:

```bash
make docker-build
make docker-up
```

Services map to `http://localhost:8000` (API) and `http://localhost:3000` (Instrument).
