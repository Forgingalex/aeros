import React, { useCallback, useEffect, useRef, useState } from "react";
import TacticalHUD from "./components/TacticalHUD";

const WS_RECONNECT_DELAY_MS = 3000;
const DEFAULT_RUNTIME_CONFIG = {
  runtime_mode: "legacy",
  safety_mode: "legacy",
  stream_view: "auto",
  predict_horizon_ms: 100,
};
const DEFAULT_PID_GAINS = { kp: 1.0, ki: 0.1, kd: 0.5 };
const DEFAULT_TELEMETRY = {
  seqId: 0,
  headingRadians: 0,
  headingDegrees: 0,
  headingRate: 0,
  controlOutput: 0,
  telemetryFps: 0,
  viewFps: 0,
  latencyMs: 0,
  motionEnergy: null,
  confidence: null,
  runtimeMode: "legacy",
  authoritativePath: "legacy",
  frameKind: "rgb",
  warmup: true,
  predictHorizonMs: 100,
  droneState: null,
  shadow: null,
  lastFrameTs: 0,
};

function toFiniteNumber(value, fallback = 0) {
  const next = Number(value);
  return Number.isFinite(next) ? next : fallback;
}

function normalizeHeadingDegrees(value) {
  const normalized = value % 360;
  return normalized < 0 ? normalized + 360 : normalized;
}

function radiansToDegrees(value) {
  return (value * 180) / Math.PI;
}

function formatApiBaseUrl() {
  const { protocol, hostname, port } = window.location;
  if (port === "3000") {
    return `${protocol}//${hostname}:8000`;
  }
  return `${protocol}//${hostname}${port ? `:${port}` : ""}`;
}

function formatWebSocketUrl() {
  const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
  const { hostname, port } = window.location;
  if (port === "3000") {
    return `${protocol}//${hostname}:8000/ws`;
  }
  return `${protocol}//${hostname}${port ? `:${port}` : ""}/ws`;
}

function App() {
  const apiBaseUrlRef = useRef(formatApiBaseUrl());
  const websocketUrlRef = useRef(formatWebSocketUrl());
  const websocketRef = useRef(null);
  const reconnectTimeoutRef = useRef(null);
  const frameCanvasRef = useRef(null);
  const frameContextRef = useRef(null);
  const frameDecodeInFlightRef = useRef(false);
  const pendingFrameBlobRef = useRef(null);
  const streamFpsWindowRef = useRef({ count: 0, startedAt: 0 });
  const telemetryRef = useRef({ ...DEFAULT_TELEMETRY });
  const runtimeConfigRef = useRef({ ...DEFAULT_RUNTIME_CONFIG });

  const [status, setStatus] = useState("connecting");
  const [error, setError] = useState(null);
  const [modelLoaded, setModelLoaded] = useState(false);
  const [running, setRunning] = useState(false);
  const [runtimeConfig, setRuntimeConfig] = useState(DEFAULT_RUNTIME_CONFIG);
  const [runtimeConfigLoading, setRuntimeConfigLoading] = useState(false);
  const [pidGains, setPidGains] = useState(DEFAULT_PID_GAINS);
  const [pidGainsLoading, setPidGainsLoading] = useState(false);
  const [cameraIndex, setCameraIndex] = useState(0);

  // Resize canvas to match the window viewport size
  const handleResize = useCallback(() => {
    const canvas = frameCanvasRef.current;
    if (canvas) {
      canvas.width = window.innerWidth;
      canvas.height = window.innerHeight;
    }
  }, []);

  useEffect(() => {
    window.addEventListener("resize", handleResize);
    handleResize(); // Initial call
    return () => {
      window.removeEventListener("resize", handleResize);
    };
  }, [handleResize]);

  const drawFrame = useCallback((bitmap) => {
    const canvas = frameCanvasRef.current;
    if (!canvas) {
      if (typeof bitmap.close === "function") {
        bitmap.close();
      }
      return;
    }

    const context =
      frameContextRef.current ||
      canvas.getContext("2d", {
        alpha: false,
        desynchronized: true,
      });

    if (!context) {
      if (typeof bitmap.close === "function") {
        bitmap.close();
      }
      return;
    }

    frameContextRef.current = context;

    // Direct width/height verification before draw to prevent visual scaling bugs
    if (canvas.width !== window.innerWidth || canvas.height !== window.innerHeight) {
      canvas.width = window.innerWidth;
      canvas.height = window.innerHeight;
    }

    // Cover-fit rendering logic (crop and center to fill viewport)
    const scale = Math.max(canvas.width / bitmap.width, canvas.height / bitmap.height);
    const dw = bitmap.width * scale;
    const dh = bitmap.height * scale;
    const dx = (canvas.width - dw) / 2;
    const dy = (canvas.height - dh) / 2;

    context.drawImage(bitmap, dx, dy, dw, dh);

    const now = performance.now();
    const streamWindow = streamFpsWindowRef.current;
    if (streamWindow.startedAt === 0) {
      streamWindow.startedAt = now;
    }

    streamWindow.count += 1;
    if (now - streamWindow.startedAt >= 1000) {
      telemetryRef.current.viewFps =
        (streamWindow.count * 1000) / (now - streamWindow.startedAt);
      streamWindow.count = 0;
      streamWindow.startedAt = now;
    }

    telemetryRef.current.lastFrameTs = now;

    if (typeof bitmap.close === "function") {
      bitmap.close();
    }
  }, []);

  const decodeFrameBlob = useCallback(async (blob) => {
    if ("createImageBitmap" in window) {
      return window.createImageBitmap(blob);
    }

    return new Promise((resolve, reject) => {
      const url = URL.createObjectURL(blob);
      const image = new Image();

      image.onload = () => {
        URL.revokeObjectURL(url);
        resolve(image);
      };

      image.onerror = () => {
        URL.revokeObjectURL(url);
        reject(new Error("Image decode failed"));
      };

      image.src = url;
    });
  }, []);

  const processQueuedFrame = useCallback(
    async (blob) => {
      frameDecodeInFlightRef.current = true;

      try {
        const decodedFrame = await decodeFrameBlob(blob);
        drawFrame(decodedFrame);
      } catch (decodeError) {
        console.error("Frame decode error:", decodeError);
      } finally {
        frameDecodeInFlightRef.current = false;
        const nextBlob = pendingFrameBlobRef.current;
        pendingFrameBlobRef.current = null;

        if (nextBlob) {
          processQueuedFrame(nextBlob);
        }
      }
    },
    [decodeFrameBlob, drawFrame]
  );

  const handleFrameBlob = useCallback(
    (blob) => {
      if (frameDecodeInFlightRef.current) {
        pendingFrameBlobRef.current = blob;
        return;
      }
      processQueuedFrame(blob);
    },
    [processQueuedFrame]
  );

  const handleTelemetryPacket = useCallback((payload) => {
    if (payload.type !== "telemetry") {
      if (payload.error) {
        setError(payload.error);
      }
      return;
    }

    const headingRadians = toFiniteNumber(payload.heading, 0);
    const headingDegrees = normalizeHeadingDegrees(radiansToDegrees(headingRadians));

    telemetryRef.current = {
      ...telemetryRef.current,
      seqId: toFiniteNumber(payload.seq_id, telemetryRef.current.seqId),
      headingRadians,
      headingDegrees,
      headingRate: toFiniteNumber(payload.heading_rate, 0),
      controlOutput: toFiniteNumber(payload.control_output, 0),
      telemetryFps: toFiniteNumber(payload.fps, 0),
      latencyMs: toFiniteNumber(payload.latency_ms, 0),
      motionEnergy:
        payload.motion_energy == null ? null : toFiniteNumber(payload.motion_energy, 0),
      confidence:
        payload.confidence == null ? null : toFiniteNumber(payload.confidence, 0),
      runtimeMode: payload.runtime_mode || "legacy",
      authoritativePath: payload.authoritative_path || "legacy",
      frameKind: payload.frame_kind || "rgb",
      warmup: Boolean(payload.warmup),
      predictHorizonMs: toFiniteNumber(
        payload.predict_horizon_ms,
        runtimeConfigRef.current.predict_horizon_ms
      ),
      droneState: payload.drone_state ?? null,
      shadow: payload.shadow ?? null,
    };
  }, []);

  useEffect(() => {
    let mounted = true;

    const connectWebSocket = () => {
      if (!mounted) return;

      setStatus("connecting");
      const socket = new WebSocket(websocketUrlRef.current);
      socket.binaryType = "blob";
      websocketRef.current = socket;

      socket.onopen = () => {
        if (!mounted) return;
        setStatus("connected");
        setError(null);
      };

      socket.onmessage = (event) => {
        if (event.data instanceof Blob) {
          handleFrameBlob(event.data);
          return;
        }

        try {
          const payload = JSON.parse(event.data);
          handleTelemetryPacket(payload);
        } catch (parseError) {
          console.error("Failed to parse websocket payload:", parseError);
        }
      };

      socket.onerror = () => {
        if (!mounted) return;
        setError("WebSocket connection error");
      };

      socket.onclose = () => {
        if (!mounted) return;
        setStatus("disconnected");
        reconnectTimeoutRef.current = window.setTimeout(
          connectWebSocket,
          WS_RECONNECT_DELAY_MS
        );
      };
    };

    connectWebSocket();

    return () => {
      mounted = false;
      if (reconnectTimeoutRef.current) {
        window.clearTimeout(reconnectTimeoutRef.current);
      }
      if (websocketRef.current) {
        websocketRef.current.close();
      }
    };
  }, [handleFrameBlob, handleTelemetryPacket]);

  const fetchRuntimeConfig = useCallback(async () => {
    try {
      const response = await fetch(`${apiBaseUrlRef.current}/runtime_config`);
      const payload = await response.json();

      if (payload.status === "success") {
        runtimeConfigRef.current = payload.config;
        setRuntimeConfig(payload.config);
      }
    } catch (requestError) {
      console.error("Failed to fetch runtime config:", requestError);
    }
  }, []);

  const handleSetCameraSource = useCallback(async (index) => {
    try {
      const response = await fetch(
        `${apiBaseUrlRef.current}/set_camera_source?index=${index}`,
        { method: "POST" }
      );
      const payload = await response.json();
      if (payload.status === "success") {
        setCameraIndex(index);
        setError(null);
      } else {
        setError(payload.message || "Failed to switch camera");
      }
    } catch (requestError) {
      console.error("Failed to switch camera:", requestError);
      setError(`Failed to switch camera: ${requestError.message}`);
    }
  }, []);

  const fetchPidGains = useCallback(async () => {
    try {
      const response = await fetch(`${apiBaseUrlRef.current}/get_pid_gains`);
      const payload = await response.json();

      if (payload.status === "success") {
        setPidGains(payload.gains);
      }
    } catch (requestError) {
      console.error("Failed to fetch PID gains:", requestError);
    }
  }, []);

  useEffect(() => {
    fetchRuntimeConfig();
    fetchPidGains();
  }, [fetchPidGains, fetchRuntimeConfig]);

  const handleSetRuntimeConfig = useCallback(async (patch) => {
    const nextConfig = {
      ...runtimeConfigRef.current,
      ...patch,
    };

    setRuntimeConfigLoading(true);

    try {
      const response = await fetch(`${apiBaseUrlRef.current}/runtime_config`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(nextConfig),
      });
      const payload = await response.json();

      if (payload.status === "success") {
        runtimeConfigRef.current = payload.config;
        setRuntimeConfig(payload.config);
        setError(null);
      } else {
        setError(payload.message || "Failed to update runtime configuration");
      }
    } catch (requestError) {
      console.error("Failed to update runtime config:", requestError);
      setError(`Failed to update runtime config: ${requestError.message}`);
    } finally {
      setRuntimeConfigLoading(false);
    }
  }, []);

  const handleLoadModel = useCallback(async () => {
    try {
      const response = await fetch(
        `${apiBaseUrlRef.current}/load_model?model_path=checkpoints/best_model.pth&use_onnx=false`,
        {
          method: "POST",
        }
      );
      const payload = await response.json();

      if (payload.status === "success") {
        setModelLoaded(true);
        setError(null);
      } else {
        setError(payload.message || "Failed to load model");
      }
    } catch (requestError) {
      console.error("Failed to load model:", requestError);
      setError(`Failed to load model: ${requestError.message}`);
    }
  }, []);

  const handleStartSimulation = useCallback(async () => {
    try {
      const response = await fetch(
        `${apiBaseUrlRef.current}/start_simulation?gui=true`,
        {
          method: "POST",
        }
      );
      const payload = await response.json();

      if (payload.status === "success") {
        setRunning(true);
        setError(null);
      } else {
        setRunning(false);
        setError(payload.message || "Failed to start simulation");
      }
    } catch (requestError) {
      console.error("Failed to start simulation:", requestError);
      setRunning(false);
      setError(`Failed to start simulation: ${requestError.message}`);
    }
  }, []);

  const handleStopSimulation = useCallback(async () => {
    try {
      const response = await fetch(`${apiBaseUrlRef.current}/stop_simulation`, {
        method: "POST",
      });
      const payload = await response.json();

      if (payload.status === "success") {
        setRunning(false);
        setError(null);
      } else {
        setError(payload.message || "Failed to stop simulation");
      }
    } catch (requestError) {
      console.error("Failed to stop simulation:", requestError);
      setError(`Failed to stop simulation: ${requestError.message}`);
    }
  }, []);

  const handleUpdatePidGains = useCallback(async (gains) => {
    setPidGainsLoading(true);

    try {
      const response = await fetch(`${apiBaseUrlRef.current}/set_pid_gains`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(gains),
      });
      const payload = await response.json();

      if (payload.status === "success") {
        setPidGains(payload.gains);
        setError(null);
      } else {
        setError(payload.message || "Failed to update PID gains");
      }
    } catch (requestError) {
      console.error("Failed to update PID gains:", requestError);
      setError(`Failed to update PID gains: ${requestError.message}`);
    } finally {
      setPidGainsLoading(false);
    }
  }, []);

  const handleResetPidGains = useCallback(async () => {
    await handleUpdatePidGains(DEFAULT_PID_GAINS);
  }, [handleUpdatePidGains]);

  return (
    <TacticalHUD
      status={status}
      error={error}
      modelLoaded={modelLoaded}
      running={running}
      runtimeConfig={runtimeConfig}
      runtimeConfigLoading={runtimeConfigLoading}
      pidGains={pidGains}
      pidGainsLoading={pidGainsLoading}
      telemetryRef={telemetryRef}
      viewportCanvasRef={frameCanvasRef}
      onLoadModel={handleLoadModel}
      onStartSimulation={handleStartSimulation}
      onStopSimulation={handleStopSimulation}
      onSetRuntimeConfig={handleSetRuntimeConfig}
      onUpdatePidGains={handleUpdatePidGains}
      onResetPidGains={handleResetPidGains}
      cameraIndex={cameraIndex}
      onSetCameraSource={handleSetCameraSource}
    />
  );
}

export default App;
