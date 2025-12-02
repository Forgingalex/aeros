import React, { useState, useEffect, useRef, useCallback } from 'react';
import './App.css';
import { AerosDashboard } from './components/AerosDashboard';

function App() {
  // Connection state
  const [status, setStatus] = useState("disconnected"); // "connected" | "disconnected" | "connecting"
  const [error, setError] = useState(null);
  
  // Model and pipeline state
  const [modelLoaded, setModelLoaded] = useState(false);
  const [running, setRunning] = useState(false);
  
  // Telemetry data stored in refs for performance (avoid re-renders)
  const telemetryRef = useRef({
    heading: 0.0,
    controlOutput: 0.0,
    fps: 0.0,
    latency: 0.0,
    frameCount: 0,
    droneState: null,
  });
  
  // State for UI updates (debounced/throttled)
  const [displayHeading, setDisplayHeading] = useState(0.0);
  const [displayControlOutput, setDisplayControlOutput] = useState(0.0);
  const [displayFps, setDisplayFps] = useState(0.0);
  const [displayLatency, setDisplayLatency] = useState(0.0);
  const [displayFrameCount, setDisplayFrameCount] = useState(0);
  const [cameraUrl, setCameraUrl] = useState(undefined);
  
  // Refs for video rendering
  const canvasRef = useRef(null);
  const wsRef = useRef(null);
  const frameCountRef = useRef(0);
  const lastTimeRef = useRef(Date.now());
  const animationFrameRef = useRef(null);
  const lastUpdateTimeRef = useRef(0);
  
  // Performance: Update UI at max 30fps (33ms intervals)
  const UI_UPDATE_INTERVAL = 33;
  
  // Update display values from refs (throttled)
  const updateDisplay = useCallback(() => {
    const now = Date.now();
    if (now - lastUpdateTimeRef.current < UI_UPDATE_INTERVAL) {
      return;
    }
    lastUpdateTimeRef.current = now;
    
    const telemetry = telemetryRef.current;
    setDisplayHeading(telemetry.heading);
    setDisplayControlOutput(telemetry.controlOutput);
    setDisplayFps(telemetry.fps);
    setDisplayLatency(telemetry.latency);
    setDisplayFrameCount(telemetry.frameCount);
  }, []);
  
  // Use requestAnimationFrame for smooth UI updates
  useEffect(() => {
    const updateLoop = () => {
      updateDisplay();
      animationFrameRef.current = requestAnimationFrame(updateLoop);
    };
    animationFrameRef.current = requestAnimationFrame(updateLoop);
    
    return () => {
      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current);
      }
    };
  }, [updateDisplay]);
  
  // WebSocket connection
  useEffect(() => {
    connectWebSocket();
    return () => {
      if (wsRef.current) {
        wsRef.current.close();
      }
    };
  }, []);

  const connectWebSocket = () => {
    setStatus("connecting");
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${protocol}//${window.location.host.replace(':3000', ':8000')}/ws`;
    
    const ws = new WebSocket(wsUrl);
    wsRef.current = ws;

    ws.onopen = () => {
      console.log('WebSocket connected');
      setStatus("connected");
      setError(null);
    };

    ws.onmessage = (event) => {
      // Check if message is JSON (telemetry) or binary (frame)
      if (event.data instanceof Blob) {
        // Binary data - frame
        handleFrameData(event.data);
      } else {
        // JSON data - telemetry
        handleTelemetryData(event.data);
      }
    };

    ws.onerror = (error) => {
      console.error('WebSocket error:', error);
      setError('WebSocket connection error');
      setStatus("disconnected");
    };

    ws.onclose = () => {
      console.log('WebSocket disconnected');
      setStatus("disconnected");
      // Attempt to reconnect after 3 seconds if not manually closed
      setTimeout(() => {
        if (!wsRef.current || wsRef.current.readyState === WebSocket.CLOSED) {
          connectWebSocket();
        }
      }, 3000);
    };
  };
  
  // Optimized frame handling with downscaling
  const handleFrameData = useCallback((blob) => {
    blob.arrayBuffer().then(buffer => {
      createImageBitmap(new Blob([buffer], { type: 'image/jpeg' }))
        .then(bitmap => {
          if (!canvasRef.current) return;
          
          const canvas = canvasRef.current;
          const ctx = canvas.getContext('2d');
          
          // Downscale for performance (max 1280px width)
          const maxWidth = 1280;
          const scale = bitmap.width > maxWidth ? maxWidth / bitmap.width : 1;
          canvas.width = bitmap.width * scale;
          canvas.height = bitmap.height * scale;
          
          ctx.drawImage(bitmap, 0, 0, canvas.width, canvas.height);
          const dataUrl = canvas.toDataURL('image/jpeg', 0.85);
          setCameraUrl(dataUrl);
          bitmap.close();
          
          // Calculate FPS
          frameCountRef.current += 1;
          const now = Date.now();
          const elapsed = (now - lastTimeRef.current) / 1000;
          if (elapsed >= 1.0) {
            const currentFps = frameCountRef.current / elapsed;
            telemetryRef.current.fps = currentFps;
            frameCountRef.current = 0;
            lastTimeRef.current = now;
          }
        })
        .catch(err => console.error('Image decode error:', err));
    });
  }, []);
  
  // Optimized telemetry handling (store in ref, update display separately)
  const handleTelemetryData = useCallback((data) => {
    try {
      const parsed = JSON.parse(data);
      if (parsed.type === 'telemetry') {
        // Update refs immediately (no re-render)
        telemetryRef.current.heading = parsed.heading || 0.0;
        telemetryRef.current.controlOutput = parsed.control_output || 0.0;
        telemetryRef.current.latency = parsed.latency_ms || 0.0;
        telemetryRef.current.frameCount = parsed.frame_count || 0;
        if (parsed.drone_state) {
          telemetryRef.current.droneState = parsed.drone_state;
        }
      } else if (parsed.error) {
        setError(parsed.error);
      }
    } catch (e) {
      console.error('Failed to parse telemetry:', e);
    }
  }, []);

  // API call functions
  const handleLoadModel = useCallback(async () => {
    try {
      const response = await fetch('http://localhost:8000/load_model?model_path=checkpoints/best_model.pth&use_onnx=false', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
      });
      const data = await response.json();
      if (data.status === 'success') {
        setModelLoaded(true);
        setError(null);
        console.log('Model loaded successfully');
      } else {
        setError(data.message || 'Failed to load model');
      }
    } catch (err) {
      console.error('Error loading model:', err);
      setError('Failed to load model: ' + err.message);
    }
  }, []);

  const handleStart = useCallback(async () => {
    try {
      const response = await fetch('http://localhost:8000/start_simulation?gui=true', {
        method: 'POST',
      });
      const data = await response.json();
      if (data.status === 'success') {
        setRunning(true);
        console.log('Simulation started');
      } else {
        setError(data.message || 'Failed to start simulation');
      }
    } catch (err) {
      console.error('Error starting simulation:', err);
      setError('Failed to start simulation');
    }
  }, []);

  const handleStop = useCallback(async () => {
    try {
      const response = await fetch('http://localhost:8000/stop_simulation', {
        method: 'POST',
      });
      const data = await response.json();
      if (data.status === 'success') {
        setRunning(false);
        console.log('Simulation stopped');
      } else {
        setError(data.message || 'Failed to stop simulation');
      }
    } catch (err) {
      console.error('Error stopping simulation:', err);
      setError('Failed to stop simulation');
    }
  }, []);

  // Convert heading from radians to degrees
  const headingDeg = (displayHeading * 180) / Math.PI;

  return (
    <>
      <canvas ref={canvasRef} style={{ display: 'none' }} />
      <AerosDashboard
        status={status}
        cameraUrl={cameraUrl}
        headingDeg={headingDeg}
        controlOutput={displayControlOutput}
        fps={displayFps}
        latencyMs={displayLatency}
        framesProcessed={displayFrameCount}
        modelLoaded={modelLoaded}
        running={running}
        onLoadModel={handleLoadModel}
        onStart={handleStart}
        onStop={handleStop}
      />
    </>
  );
}

export default App;
