import React, { useState, useEffect, useRef } from 'react';
import './App.css';
import Logo from './components/Logo';
import MetricCard from './components/MetricCard';
import HeadingIndicator from './components/HeadingIndicator';

function App() {
  const [connected, setConnected] = useState(false);
  const [heading, setHeading] = useState(0.0);
  const [controlOutput, setControlOutput] = useState(0.0);
  const [fps, setFps] = useState(0.0);
  const [latency, setLatency] = useState(0.0);
  const [frameCount, setFrameCount] = useState(0);
  const [droneState, setDroneState] = useState(null);
  const [error, setError] = useState(null);
  
  const videoRef = useRef(null);
  const wsRef = useRef(null);
  const frameCountRef = useRef(0);
  const lastTimeRef = useRef(Date.now());

  useEffect(() => {
    connectWebSocket();
    return () => {
      if (wsRef.current) {
        wsRef.current.close();
      }
    };
  }, []);

  const connectWebSocket = () => {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${protocol}//${window.location.host.replace(':3000', ':8000')}/ws`;
    
    const ws = new WebSocket(wsUrl);
    wsRef.current = ws;

    ws.onopen = () => {
      console.log('WebSocket connected');
      setConnected(true);
      setError(null);
    };

    ws.onmessage = (event) => {
      // Check if message is JSON (telemetry) or binary (frame)
      if (event.data instanceof Blob) {
        // Binary data - frame
        const reader = new FileReader();
        reader.onload = () => {
          const blob = new Blob([reader.result], { type: 'image/jpeg' });
          const url = URL.createObjectURL(blob);
          if (videoRef.current) {
            videoRef.current.src = url;
            // Clean up previous URL
            if (videoRef.current.dataset.prevUrl) {
              URL.revokeObjectURL(videoRef.current.dataset.prevUrl);
            }
            videoRef.current.dataset.prevUrl = url;
          }
          
          // Calculate FPS
          frameCountRef.current += 1;
          const now = Date.now();
          const elapsed = (now - lastTimeRef.current) / 1000;
          if (elapsed >= 1.0) {
            const currentFps = frameCountRef.current / elapsed;
            setFps(currentFps);
            frameCountRef.current = 0;
            lastTimeRef.current = now;
          }
        };
        reader.readAsArrayBuffer(event.data);
      } else {
        // JSON data - telemetry
        try {
          const data = JSON.parse(event.data);
          if (data.type === 'telemetry') {
            setHeading(data.heading || 0.0);
            setControlOutput(data.control_output || 0.0);
            setLatency(data.latency_ms || 0.0);
            setFrameCount(data.frame_count || 0);
            if (data.drone_state) {
              setDroneState(data.drone_state);
            }
          } else if (data.error) {
            setError(data.error);
          }
        } catch (e) {
          console.error('Failed to parse telemetry:', e);
        }
      }
    };

    ws.onerror = (error) => {
      console.error('WebSocket error:', error);
      setError('WebSocket connection error');
      setConnected(false);
    };

    ws.onclose = () => {
      console.log('WebSocket disconnected');
      setConnected(false);
      // Attempt to reconnect after 3 seconds
      setTimeout(() => {
        if (!connected) {
          connectWebSocket();
        }
      }, 3000);
    };
  };

  const formatAngle = (radians) => {
    const degrees = (radians * 180) / Math.PI;
    return degrees.toFixed(2);
  };

  return (
    <div className="App">
      <header className="App-header">
        <div className="header-left">
          <Logo size={40} />
          <div className="header-title">
            <h1>AEROS</h1>
            <span className="header-subtitle">Autonomy Pipeline</span>
          </div>
        </div>
        <div className="header-right">
          <div className={`status-indicator ${connected ? 'connected' : 'disconnected'}`}>
            <span className="status-dot"></span>
            <span className="status-text">{connected ? 'Connected' : 'Disconnected'}</span>
          </div>
        </div>
      </header>

      <main className="dashboard">
        {error && (
          <div className="error-banner">
            Error: {error}
          </div>
        )}

        <div className="dashboard-grid">
          {/* Camera View */}
          <div className="panel camera-panel">
            <div className="panel-header">
              <h2>
                <span className="panel-icon">📹</span>
                Camera Feed
              </h2>
            </div>
            <div className="video-container">
              <img
                ref={videoRef}
                alt="Camera feed"
                className="camera-feed"
              />
              {!connected && (
                <div className="video-placeholder">
                  <div className="placeholder-icon">📡</div>
                  <div className="placeholder-text">Waiting for connection...</div>
                  <div className="placeholder-subtext">Ensure API server is running</div>
                </div>
              )}
              {connected && (
                <div className="video-overlay">
                  <div className="overlay-badge">
                    <span className="badge-dot"></span>
                    LIVE
                  </div>
                </div>
              )}
            </div>
          </div>

          {/* Telemetry Panel */}
          <div className="panel telemetry-panel">
            <div className="panel-header">
              <h2>
                <span className="panel-icon">📊</span>
                Telemetry
              </h2>
            </div>
            <div className="telemetry-grid">
              <div className="heading-section">
                <HeadingIndicator heading={heading} size={180} />
              </div>

              <MetricCard
                label="Control Output"
                value={controlOutput.toFixed(4)}
                unit="rad/s"
                icon="⚡"
                color="primary"
              />

              <div className="metrics-row">
                <MetricCard
                  label="FPS"
                  value={fps.toFixed(1)}
                  icon="🎯"
                  color="info"
                />
                <MetricCard
                  label="Latency"
                  value={latency.toFixed(2)}
                  unit="ms"
                  icon="⏱️"
                  color={latency < 100 ? 'success' : latency < 200 ? 'warning' : 'primary'}
                />
              </div>

              <MetricCard
                label="Frames Processed"
                value={frameCount.toLocaleString()}
                icon="🎬"
                color="primary"
              />
            </div>
          </div>

          {/* Drone State Panel */}
          {droneState && (
            <div className="panel drone-panel">
              <div className="panel-header">
                <h2>
                  <span className="panel-icon">🚁</span>
                  Drone State
                </h2>
              </div>
              <div className="drone-state">
                <MetricCard
                  label="Position"
                  value={`[${droneState.position[0].toFixed(2)}, ${droneState.position[1].toFixed(2)}, ${droneState.position[2].toFixed(2)}]`}
                  icon="📍"
                  color="primary"
                />
                <MetricCard
                  label="Heading"
                  value={formatAngle(droneState.heading)}
                  unit="°"
                  icon="🧭"
                  color="info"
                />
                <MetricCard
                  label="Velocity"
                  value={`[${droneState.velocity[0].toFixed(2)}, ${droneState.velocity[1].toFixed(2)}, ${droneState.velocity[2].toFixed(2)}]`}
                  icon="💨"
                  color="success"
                />
              </div>
            </div>
          )}
        </div>
      </main>
    </div>
  );
}

export default App;

