# AEROS

**Autonomy Pipeline for Lightweight Drone Navigation**

AEROS is a compact visual autonomy stack designed to process real-time camera input, estimate heading, and execute stable navigation commands within a simulated or lightweight drone environment.

## Overview

AEROS demonstrates a complete autonomy pipeline featuring:
- **Real-time video processing** (webcam or simulation feed)
- **CNN-based heading estimation** trained on synthetic corridor data
- **PID control** for heading correction with real-time tuning
- **PyBullet simulation** environment
- **Web dashboard** with interactive PID control panel and telemetry visualization

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      AEROS Architecture                      │
└─────────────────────────────────────────────────────────────┘

┌──────────────┐         ┌──────────────┐         ┌──────────────┐
│   Camera     │────────▶│ Preprocessing│────────▶│    Model     │
│  / Sim Feed  │         │   Pipeline   │         │  (CNN/ONNX)  │
└──────────────┘         └──────────────┘         └──────┬───────┘
                                                          │
                                                          ▼
┌──────────────┐         ┌──────────────┐         ┌──────────────┐
│   React      │◀────────│   FastAPI    │◀────────│     PID      │
│  Dashboard   │  WebSocket│   Server    │         │  Controller  │
└──────────────┘         └──────────────┘         └──────┬───────┘
                                                          │
                                                          ▼
                                                  ┌──────────────┐
                                                  │  PyBullet    │
                                                  │  Simulation  │
                                                  └──────────────┘
```

### Component Overview

1. **Preprocessing** (`src/preprocessing/`)
   - Image resize, denoising, brightness normalization
   - Optimized for real-time performance

2. **Model** (`src/model/`)
   - Lightweight CNN architecture (MobileNet-style)
   - PyTorch training pipeline
   - ONNX export support

3. **Control** (`src/control/`)
   - PID controller for heading correction
   - Configurable gains and output limits

4. **Simulation** (`src/simulation/`)
   - PyBullet-based drone simulation
   - Corridor environment
   - Camera feed generation

5. **API** (`api/`)
   - FastAPI REST endpoints
   - WebSocket streaming for real-time telemetry

6. **Dashboard** (`web/`)
   - React-based web interface
   - Live camera feed
   - Real-time metrics visualization
   - Interactive PID control panel
   - Premium aviation-grade UI design

## Requirements

- Python 3.10+
- Node.js 18+ (for web dashboard)
- CUDA-capable GPU (optional, for training)

## Quick Start

### 1. Installation

```bash
# Clone repository
git clone <repository-url>
cd aeros

# Install Python dependencies
make install
# or
pip install -r requirements.txt

# Install web dependencies
cd web && npm install && cd ..
```

### 2. Generate Synthetic Data

```bash
make generate-data
# or
python scripts/generate_synthetic_data.py --output-dir data/synthetic --num-samples 10000
```

### 3. Train Model

```bash
# Option 1: Use Jupyter notebook
jupyter notebook notebooks/training.ipynb

# Option 2: Run training script (if implemented)
python scripts/train.py
```

### 4. Export ONNX Model (Optional)

```bash
make export-onnx
# or
python scripts/export_onnx.py --checkpoint models/checkpoints/best_model.pth --output models/heading_model.onnx
```

### 5. Run Backend API

```bash
make run-api
# or
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

### 6. Run Web Dashboard

In a separate terminal:

```bash
make run-web
# or
cd web && npm start
```

The dashboard will be available at `http://localhost:3000`

## Docker Deployment

### Build and Run

```bash
# Build images
make docker-build

# Start services
make docker-up

# Stop services
make docker-down
```

Services will be available at:
- API: `http://localhost:8000`
- Dashboard: `http://localhost:3000`

## Usage Guide

### Starting the Simulation

1. Start the API server:
   ```bash
   make run-api
   ```

2. Load a trained model:
   ```bash
   curl -X POST "http://localhost:8000/load_model" \
     -H "Content-Type: application/json" \
     -d '{"model_path": "models/checkpoints/best_model.pth", "use_onnx": false}'
   ```

3. Start simulation:
   ```bash
   curl -X POST "http://localhost:8000/start_simulation?gui=true"
   ```

4. Open dashboard at `http://localhost:3000` and connect via WebSocket

### Using Webcam Instead

The system will automatically use your webcam if simulation is not started. Ensure your camera is accessible and run:

```bash
make run-api
```

Then open the dashboard to see the live feed.

### API Endpoints

- `GET /` - API information
- `GET /health` - Health check
- `POST /load_model` - Load inference model
  ```json
  {
    "model_path": "models/checkpoints/best_model.pth",
    "use_onnx": false
  }
  ```
- `POST /start_simulation?gui=true` - Start PyBullet simulation
- `POST /stop_simulation` - Stop simulation
- `GET /get_pid_gains` - Get current PID controller gains
  ```json
  {
    "status": "success",
    "gains": {
      "kp": 1.0,
      "ki": 0.1,
      "kd": 0.5
    }
  }
  ```
- `POST /set_pid_gains` - Update PID controller gains
  ```json
  {
    "kp": 1.5,
    "ki": 0.15,
    "kd": 0.8
  }
  ```
- `WS /ws` - WebSocket stream for telemetry and frames

## Testing

```bash
make test
# or
pytest tests/ -v --cov=src
```

## Project Structure

```
aeros/
├── src/                    # Core Python modules
│   ├── preprocessing/      # Image preprocessing
│   ├── model/              # CNN model and training
│   ├── control/            # PID controller
│   ├── simulation/         # PyBullet simulation
│   └── utils/              # Utilities (inference, metrics)
├── api/                    # FastAPI backend
│   ├── main.py            # Main API server
│   └── websocket.py       # WebSocket utilities
├── web/                    # React dashboard
│   ├── src/
│   └── public/
├── notebooks/              # Jupyter notebooks
│   └── training.ipynb     # Training notebook
├── scripts/                # Utility scripts
│   ├── generate_synthetic_data.py
│   └── export_onnx.py
├── tests/                  # Unit tests
├── data/                   # Data directory
│   └── synthetic/         # Synthetic dataset
├── models/                 # Model checkpoints
│   └── checkpoints/
├── Dockerfile              # Docker image definition
├── docker-compose.yml      # Docker Compose config
├── Makefile               # Build automation
├── requirements.txt       # Python dependencies
└── README.md              # This file
```

## Model Training

### Dataset Format

The synthetic dataset consists of:
- Images: PNG files (224x224 RGB)
- Metadata: JSON files with image paths and heading angles

Example metadata entry:
```json
{
  "image_path": "train/image_000001.png",
  "heading": 0.1234
}
```

### Training Process

1. Generate synthetic data (see Quick Start)
2. Open `notebooks/training.ipynb` in Jupyter
3. Run all cells to train the model
4. Checkpoints are saved to `models/checkpoints/`
5. Best model is saved as `best_model.pth`

### Model Architecture

- **Input**: 224x224 RGB images
- **Architecture**: MobileNet-style depthwise separable convolutions
- **Output**: Single scalar (heading angle in radians)
- **Parameters**: ~500K-1M (lightweight for edge deployment)

## Configuration

### PID Controller Tuning

The PID controller can be tuned in two ways:

#### Method 1: Dashboard UI (Recommended)

1. Start the dashboard and load the model
2. Navigate to the **PID Controller** panel
3. Use the interactive controls:
   - **Presets**: Click preset buttons (Default, Aggressive, Smooth, Fast Response, Stable)
   - **Manual Adjustment**: Use sliders or number inputs to adjust Kp, Ki, Kd
   - **Apply Changes**: Click to update gains in real-time
   - **Reset**: Restore default values (Kp=1.0, Ki=0.1, Kd=0.5)

#### Method 2: API Endpoints

**Get current gains:**
```bash
curl http://localhost:8000/get_pid_gains
```

**Update gains:**
```bash
curl -X POST "http://localhost:8000/set_pid_gains" \
  -H "Content-Type: application/json" \
  -d '{"kp": 1.5, "ki": 0.15, "kd": 0.8}'
```

#### Method 3: Code Configuration

Edit `api/main.py` to set default gains:

```python
pid_controller = PIDController(
    kp=1.0,   # Proportional gain
    ki=0.1,   # Integral gain
    kd=0.5,   # Derivative gain
)
```

#### PID Tuning Guide

**Understanding PID Gains:**

- **Kp (Proportional)**: Controls response speed
  - Higher Kp = Faster response, but may cause overshoot
  - Lower Kp = Slower response, more stable
  - Default: 1.0

- **Ki (Integral)**: Eliminates steady-state error
  - Higher Ki = Better accuracy, but may cause oscillation
  - Lower Ki = May have steady-state error
  - Default: 0.1

- **Kd (Derivative)**: Reduces overshoot and oscillation
  - Higher Kd = More damping, less overshoot
  - Lower Kd = Less damping, may oscillate
  - Default: 0.5

**Preset Configurations:**

- **Default** (Kp=1.0, Ki=0.1, Kd=0.5): Balanced performance
- **Aggressive** (Kp=2.0, Ki=0.2, Kd=1.0): Fast response, may overshoot
- **Smooth** (Kp=0.5, Ki=0.05, Kd=0.3): Slow, stable response
- **Fast Response** (Kp=1.5, Ki=0.15, Kd=0.8): Quick correction with damping
- **Stable** (Kp=0.8, Ki=0.2, Kd=0.6): High stability, good accuracy

**Tuning Tips:**

1. Start with Default preset and observe behavior
2. If response is too slow, increase Kp gradually
3. If there's overshoot, increase Kd
4. If steady-state error exists, increase Ki slightly
5. Test one gain at a time to understand its effect
6. Watch the "Control output" metric for immediate feedback
7. Use Reset button to return to defaults if system becomes unstable

### Simulation Parameters

Edit `src/simulation/drone_sim.py` to adjust:
- Corridor dimensions
- Camera parameters
- Drone physics

## Performance Targets

- **Inference**: ≥ 15 FPS on CPU
- **Latency**: < 100ms end-to-end
- **Model Size**: < 10MB (ONNX)
- **Accuracy**: MAE < 0.1 radians on validation set

## Troubleshooting

### WebSocket Connection Issues

- Ensure API server is running on port 8000
- Check CORS settings if accessing from different origin
- Verify firewall settings

### Simulation Not Starting

- Ensure PyBullet is installed: `pip install pybullet`
- Check for display/GUI issues (use `gui=false` for headless)
- Verify OpenGL drivers are installed

### Model Loading Errors

- Check model path is correct
- Ensure checkpoint file exists
- Verify model architecture matches checkpoint

## Known Limitations (v0.1.0-alpha)

This is a pre-release version. The following limitations should be considered:

- **Simulation Physics**: Simplified torque-based control, not full aerodynamics. Real-world behavior may differ.
- **Training Data**: Model trained on synthetic corridor data only. Performance on real-world images may vary.
- **Hardware Access**: Webcam access in Docker requires privileged mode and device mapping (`/dev/video0`).
- **Optional Dependencies**: PyBullet and ONNX Runtime are optional (system degrades gracefully if not installed).
- **Real-world Testing**: Not yet tested on physical drones. Use with caution in production environments.
- **Model Accuracy**: Performance may vary with different lighting conditions, camera angles, and environments.
- **PID Tuning**: Controller gains are tuned for simulation. May require retuning for real hardware.
- **Single Drone**: Currently supports single drone simulation. Multi-drone support planned for future releases.

For the latest updates and known issues, see [CHANGELOG.md](CHANGELOG.md).

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Contact

For questions or issues, please open an issue on GitHub.

---

**Built with**: PyTorch, FastAPI, React, PyBullet

Aeros it is...

