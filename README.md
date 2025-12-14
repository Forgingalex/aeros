# AEROS — Autonomous Edge Robotics Operating System

A real-time, camera-only autonomy pipeline built to treat latency as a first-class design constraint.

---

## The Constraint That Matters

Latency kills control stability. Not model accuracy, not sensor fusion complexity—latency.

When your perception loop runs at 30 FPS and your control loop needs corrections every 33ms, a 50ms preprocessing step means you're always one frame behind. The PID controller sees stale heading estimates. The drone oscillates. You debug for hours thinking it's a tuning problem when it's actually a timing problem.

Edge hardware makes this worse. A Raspberry Pi 4 can't run ResNet in real-time. A Jetson Nano can, but only if you're careful about memory bandwidth and quantization. Every millisecond you spend on preprocessing or inference directly couples with control behavior. There's no buffer, no queue, no graceful degradation—just missed deadlines and unstable flight.

This system exists because I needed to see how far camera-only perception can go when you treat latency as the primary constraint, not an afterthought.

---

## The Question Driving This System

**How far can camera-only perception go when every millisecond of latency directly couples with control behavior?**

Not "can we build a better SLAM system" or "can we add more sensors." Just: given a single camera, a lightweight CNN, and strict timing requirements, what's the minimum viable perception pipeline that keeps a drone stable?

The answer isn't in the model architecture. It's in the preprocessing choices, the async boundaries, the frontend rendering discipline. It's in choosing Gaussian blur over Non-Local Means because 10ms beats 200ms every time. It's in running physics at a fixed 60 Hz independent of client count because time warps break control loops.

---

## What AEROS Actually Is

A perception-to-control pipeline with strict timing guarantees:

```
Camera/Sim -> Preprocessing -> CNN Inference -> PID Control -> Actuators
                |                |              |
            <10ms            <30ms          <5ms
```

The perception side: MobileNet-style CNN (~500K-1M parameters) that predicts heading from 224x224 RGB frames. Gaussian blur and CLAHE for preprocessing. Dual inference paths (PyTorch for development, ONNX for edge deployment).

The control side: PID controller with configurable gains. Torque-based physics simulation that respects inertia and mass. Background physics loop running at fixed 60 Hz, decoupled from WebSocket streaming.

The telemetry side: FastAPI backend with WebSocket binary frame streaming. React dashboard that stores high-frequency data in refs, throttles UI updates to 30 FPS, uses `createImageBitmap` to avoid GC pressure.

Everything is instrumented. Every component reports latency. The system fails fast when deadlines are missed.

---

## Design Decisions I Would Not Undo

**Gaussian blur over Non-Local Means.** NLM gives better denoising but costs 50-200ms per frame. Gaussian blur keeps preprocessing under 10ms. The accuracy tradeoff is acceptable when latency is the constraint.

**Torque-based control over direct velocity setting.** Setting velocity directly is simpler but unrealistic. Torque-based control respects inertia, makes the simulation behave like real hardware, and forces proper PID tuning. The extra complexity is worth it.

**Fixed-rate physics loop in background task.** Multiple WebSocket clients used to cause physics time warp—simulation would speed up or slow down unpredictably. Moving physics stepping to a background `asyncio.Task` at 60 Hz fixed it. Physics time is now independent of client count.

**Frontend performance discipline.** Telemetry in refs, not state. Throttled UI updates. `createImageBitmap` instead of Blob URLs. These aren't micro-optimizations—they're the difference between a responsive dashboard and a frozen browser when streaming at 30 FPS.

**BGR->RGB conversion in inference engine.** The model was trained on RGB. OpenCV returns BGR. Without explicit conversion, the model sees inverted color channels. This was a subtle bug that degraded accuracy until I caught it.

**ONNX export support.** PyTorch is fine for development. ONNX is required for edge deployment. Dual inference paths let me develop with PyTorch and deploy with ONNX without changing code.

---

## What This System Does Not Attempt

No SLAM. No obstacle detection. No path planning. No multi-drone coordination. No ROS2 integration.

The scope is intentionally narrow: heading estimation from camera frames, PID-based heading correction, simulation environment for testing. Everything else is out of scope because it would add latency or complexity that breaks the core constraint.

The model is trained on synthetic corridor data only. Real-world performance is untested. The simulation uses simplified torque-based physics, not full aerodynamics. The PID controller is tuned for simulation—real hardware will need retuning.

These boundaries exist because the question isn't "can we build a complete autonomy stack" but "can we make camera-only perception fast enough for stable control."

---

## Where This Is Headed

Event-based visual representations. Converting frames to event streams (pixel-level brightness changes) could reduce preprocessing to <2ms. Need to test latency vs accuracy tradeoff on Jetson Nano.

Perception confidence estimation. Model uncertainty via Monte Carlo dropout or ensemble variance. Low-confidence predictions indicate ambiguous scenes where control should be damped. Measure correlation between confidence and control stability.

Confidence-conditioned control. Scale PID gains based on perception confidence. When confidence < threshold, reduce Kp/Kd to prevent overcorrection on ambiguous visual features. Test whether this reduces oscillation in low-texture environments.

Edge deployment guides. Quantization, model compression, deployment on Jetson Nano and Raspberry Pi. The ONNX path exists, but deployment workflows need documentation.

Real-world dataset collection. Synthetic data works for proof-of-concept. Real corridors have different lighting, textures, and visual features. Need to collect and train on real data to validate the approach.

---

## For the Full Engineering Details

Setup instructions, API documentation, architecture diagrams, performance targets, Docker deployment, testing—see [TECHNICAL.md](TECHNICAL.md).

Quick start:

```bash
pip install -r requirements.txt
cd web && npm install && cd ..
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
cd web && npm start
```

Dashboard at `http://localhost:3000`.

---

**Built with:** PyTorch, FastAPI, React, PyBullet

*This README reflects only what's implemented. All technical decisions are verifiable in the source code and CHANGELOG.*
...aeros it is
