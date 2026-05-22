# AEROS — Autonomous Edge Robotics Operating System

A real-time, motion-first autonomy pipeline built to treat latency as a law, not a suggestion.

<div align="center">
  <img src="https://github.com/user-attachments/assets/48f5aced-001a-46cb-b109-cd301ff7eac5" alt="AEROS Logo" width="150"/>
</div>

---

## The Constraint That Matters

Latency kills control stability. Not model accuracy, not sensor fusion complexity; latency.

When your perception loop runs at 30 FPS and your control loop needs corrections every 33ms, a 50ms preprocessing step means you are always one frame behind. The PID controller sees stale heading estimates. The drone oscillates. You debug for hours thinking it is a tuning problem when it is actually a timing problem.

This system exists to answer one question: given a single camera, a lightweight CNN, and strict timing requirements, what is the minimum viable perception pipeline that keeps a drone stable? 

I enforce a strict 5ms budget per stage. Any feature that takes longer is rejected, redesigned, or disabled.

---

## The Engine: The Authoritative Heartbeat

The original Alpha version served as foundational base, but it tied perception and control to the WebSocket client connection. I rebuilt the system for professional-grade autonomy.

Today, AEROS runs on a single **Authoritative Heartbeat**. The backend uses a dedicated background runtime loop for each active video source. This loop owns frame acquisition, inference, control computation, and telemetry generation. WebSocket subscribers are strictly passive read-only consumers. If ten clients connect, or if zero clients connect, the physics and perception loop marches on at its fixed rate. 

I also introduced **Camera Hot-Swapping**. You can switch the active sensor on the fly, from a built-in webcam to an iPhone Continuity Camera via a POST endpoint. The engine drops the old stream and initializes the new one without killing the heartbeat or severing the telemetry connection.

---

## The Instrument: Aviation-Window HUD

The frontend is a high-fidelity flight instrument that feels alive and mechanical. It completely bypasses React state for high-frequency motion.

**Aviation-Window Tapes:** The pilot's eyes must remain stationary. The digital readouts for Heading, FPS, and Energy sit in fixed glass boxes in the center of the viewport. The actual scalar tracks physically slide behind these windows. 

**Active Bore-Sight:** The center crosshair is a functional kinetic indicator. A Velocity Vector "Ghost Icon" shifts laterally based on the raw heading rate to show actual physical drift. AOA brackets pulse and scale directly with motion energy. *Note: Currently, the Velocity Vector's drift is a visual indicator driven by telemetry, not a separate predictive physics calculation.*

**Performance Law:** To stay under the 5ms latency budget, all sliding tapes and pulsing brackets update via direct DOM mutations inside a `requestAnimationFrame` loop. I use `createImageBitmap` and `useRef` to eliminate garbage collection pressure and prevent React from choking the main thread. The layout is strictly locked to 100vh/100vw. No scrolling, no reflows.

---

## What This System Does Not Attempt

No SLAM. No path planning. No multi-drone coordination. No ROS2 integration. 

The model is trained on synthetic corridor data. Real-world performance remains untested on physical drone hardware. The simulation uses simplified torque-based physics, not full aerodynamics. 

**macOS Native Simulation is Disabled:** Upstream Bullet physics currently fails to build against recent Apple SDKs. If you are running AEROS natively on macOS, the simulation endpoints will fail. The camera feed, UI, and inference engine work perfectly, but simulation requires Docker/Linux. We do not hide this flaw. 

---

## Where I'm Going (The Mandate)

I am actively transitioning to the **AEROS.vOmega** architecture, prioritizing motion over color.

**Temporal Eyes:** I will perceive geometry through change. The pipeline will compute temporal delta encodings from grayscale frames, dropping reliance on static RGB textures. 

**Predictive Brain:** I will replace the frame-wise CNN with a lightweight State-Space Model (SSM-lite) that reasons over temporal contrast and maintains a hidden state.

**Algebraic Shield:** I am prototyping a Conformal Geometric Algebra (CGA) safety layer to evaluate whether the vehicle remains inside a valid motion tube, providing mathematical certainty over branchy heuristics.

**Sim-to-Real Bridge:** I will deploy ONNX INT8 quantized models to Pi 5 edge hardware and talk to standard flight controllers via MAVLink SITL.

---

## Quick Start

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cd web && npm install && cd ..

# Terminal 1: The Heartbeat
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

# Terminal 2: The Instrument
cd web && npm start
```

For headless simulation (bypassing the macOS local build failure):
```bash
docker compose up --build api
```

The flight instrument mounts at `http://localhost:3000`.

*This documentation reflects AEROS as it exists today in the main branch.*
