import React, { useEffect, useRef } from "react";
import HeadingTape from "./HeadingTape";
import BoreSight from "./BoreSight";
import VerticalTape from "./VerticalTape";
import PIDNode from "./PIDNode";
import ModeNode from "./ModeNode";

const REMOTE_LOGO_URL =
  "https://github.com/user-attachments/assets/48f5aced-001a-46cb-b109-cd301ff7eac5";

// A boot / link diagnostic display when the video stream is cold
function ColdStateOverlay({ status }) {
  return (
    <div className="hud-cold-state-overlay">
      <div className="hud-diagnostics-box">
        <div className="hud-diagnostics-header">
          <div className="hud-diagnostics-title">AEROS Nav System</div>
          <div className={`hud-diagnostics-status hud-status-${status}`}>
            {status === "connecting" ? "LINK ACQUIRING" : "LINK STANDBY"}
          </div>
        </div>
        <div className="hud-diagnostics-body">
          <div className="hud-diagnostics-alert">
            {status === "connecting"
              ? "Reacquiring telemetry and telemetry stream over WebSocket link..."
              : "Heartbeat established. Awaiting active JPEG frames from telemetry pipeline..."}
          </div>
          <div className="hud-diagnostics-grid">
            <div className="hud-diagnostics-tile">
              <span className="hud-diagnostics-lbl">System Source</span>
              <span className="hud-diagnostics-val">Mac Camera</span>
            </div>
            <div className="hud-diagnostics-tile">
              <span className="hud-diagnostics-lbl">WS Endpoint</span>
              <span className="hud-diagnostics-val">/ws</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

// Micro Telemetry Strip at the bottom center. Driven by requestAnimationFrame.
function TelemetryStrip({ telemetryRef }) {
  const seqIdRef = useRef(null);
  const latencyRef = useRef(null);
  const controlRef = useRef(null);
  const modeRef = useRef(null);
  const frameKindRef = useRef(null);
  const confidenceRef = useRef(null);
  const requestRef = useRef(null);

  useEffect(() => {
    const updateStrip = () => {
      const telemetry = telemetryRef.current;
      if (!telemetry) {
        requestRef.current = requestAnimationFrame(updateStrip);
        return;
      }

      if (seqIdRef.current) {
        seqIdRef.current.textContent = `#${String(telemetry.seqId).padStart(5, "0")}`;
      }
      if (latencyRef.current) {
        latencyRef.current.textContent = `${telemetry.latencyMs.toFixed(1)}`;
        // Highlight high latency
        if (telemetry.latencyMs > 50) {
          latencyRef.current.className = "hud-telemetry-val err";
        } else if (telemetry.latencyMs > 30) {
          latencyRef.current.className = "hud-telemetry-val warn";
        } else {
          latencyRef.current.className = "hud-telemetry-val";
        }
      }
      if (controlRef.current) {
        controlRef.current.textContent = `${telemetry.controlOutput.toFixed(3)}`;
      }
      if (modeRef.current) {
        const mode = telemetry.runtimeMode === "omega_shadow" ? "SHADOW" : "LEGACY";
        modeRef.current.textContent = mode;
      }
      if (frameKindRef.current) {
        frameKindRef.current.textContent = String(telemetry.frameKind).toUpperCase();
      }
      if (confidenceRef.current) {
        const conf = telemetry.confidence;
        if (conf == null || conf <= 0) {
          confidenceRef.current.textContent = "DORMANT";
          confidenceRef.current.style.color = "var(--hud-text-dim)";
        } else {
          confidenceRef.current.textContent = `${conf.toFixed(2)}`;
          confidenceRef.current.style.color = "var(--hud-cyan)";
        }
      }

      requestRef.current = requestAnimationFrame(updateStrip);
    };

    requestRef.current = requestAnimationFrame(updateStrip);

    return () => {
      if (requestRef.current) {
        cancelAnimationFrame(requestRef.current);
      }
    };
  }, [telemetryRef]);

  return (
    <div className="hud-telemetry-strip">
      <div className="hud-telemetry-item">
        <span className="hud-telemetry-lbl">SEQ:</span>
        <span ref={seqIdRef} className="hud-telemetry-val">#00000</span>
      </div>
      <div className="hud-telemetry-item">
        <span className="hud-telemetry-lbl">LAT:</span>
        <span ref={latencyRef} className="hud-telemetry-val">0.0</span>
        <span className="hud-telemetry-unit">ms</span>
      </div>
      <div className="hud-telemetry-item">
        <span className="hud-telemetry-lbl">CTRL:</span>
        <span ref={controlRef} className="hud-telemetry-val">0.000</span>
        <span className="hud-telemetry-unit">rad/s</span>
      </div>
      <div className="hud-telemetry-item">
        <span className="hud-telemetry-lbl">MODE:</span>
        <span ref={modeRef} className="hud-telemetry-val">LEGACY</span>
      </div>
      <div className="hud-telemetry-item">
        <span className="hud-telemetry-lbl">KIND:</span>
        <span ref={frameKindRef} className="hud-telemetry-val">RGB</span>
      </div>
      <div className="hud-telemetry-item">
        <span className="hud-telemetry-lbl">CONF:</span>
        <span ref={confidenceRef} className="hud-telemetry-val">DORMANT</span>
      </div>
    </div>
  );
}

export default function TacticalHUD({
  status,
  error,
  modelLoaded,
  running,
  runtimeConfig,
  runtimeConfigLoading,
  pidGains,
  pidGainsLoading,
  telemetryRef,
  viewportCanvasRef,
  onLoadModel,
  onStartSimulation,
  onStopSimulation,
  onSetRuntimeConfig,
  cameraIndex,
  onSetCameraSource,
  onUpdatePidGains,
  onResetPidGains,
}) {
  const warmupBadgeRef = useRef(null);
  const requestRef = useRef(null);

  // Monitor the warmup badge state without re-rendering TacticalHUD
  useEffect(() => {
    const updateWarmup = () => {
      const telemetry = telemetryRef.current;
      if (warmupBadgeRef.current) {
        const showWarmup = telemetry ? telemetry.warmup : false;
        warmupBadgeRef.current.style.display = showWarmup ? "block" : "none";
      }
      requestRef.current = requestAnimationFrame(updateWarmup);
    };

    requestRef.current = requestAnimationFrame(updateWarmup);

    return () => {
      if (requestRef.current) {
        cancelAnimationFrame(requestRef.current);
      }
    };
  }, [telemetryRef]);

  // Determine if viewport is cold (no active frame from server)
  const isCold = status !== "connected";

  return (
    <div className="hud-root">
      {/* Absolute background viewport canvas */}
      <canvas ref={viewportCanvasRef} className="hud-viewport-canvas" />

      {/* Boot / Link Diagnostics overlay */}
      {isCold && <ColdStateOverlay status={status} />}

      {/* Floating Error Alert Banner */}
      {error && <div className="hud-alert-banner">{error}</div>}

      {/* Warmup Indicator Badge */}
      <div ref={warmupBadgeRef} className="hud-warmup-badge" style={{ display: "none" }}>
        Warmup / Passthrough
      </div>

      {/* Top Left: Logo + Brand watermark */}
      <div className="hud-node hud-brand-node">
        <div className="hud-logo-container">
          <img src={REMOTE_LOGO_URL} alt="AEROS Logo" className="hud-logo-img" />
        </div>
        <div className="hud-brand-info">
          <div className="hud-brand-name">AEROS</div>
          <div className="hud-tagline">Kinetic HUD</div>
        </div>
        
        {/* Link Status */}
        <div className="hud-status-indicator">
          <span className={`hud-status-${status} hud-status-label`}>
            <span className="hud-status-dot" />
            <span style={{ marginLeft: "6px" }}>
              {status === "connected"
                ? "LINK LIVE"
                : status === "connecting"
                ? "LINK ACQUIRING"
                : "LINK OFFLINE"}
            </span>
          </span>
        </div>
      </div>

      {/* Top Center: Horizontal Sliding Compass */}
      <HeadingTape telemetryRef={telemetryRef} />

      {/* Center: Bore-Sight SVG Crosshair */}
      <BoreSight telemetryRef={telemetryRef} />

      {/* Left Rail: FPS Tape */}
      <VerticalTape
        side="left"
        label="FPS"
        min={0}
        max={60}
        marks={[0, 10, 20, 30, 40, 50, 60]}
        precision={1}
        getValue={(t) => t.viewFps}
        telemetryRef={telemetryRef}
      />

      {/* Right Rail: Motion Energy (Dormant-aware) */}
      <VerticalTape
        side="right"
        label="Energy"
        min={0}
        max={0.2}
        marks={[0, 0.04, 0.08, 0.12, 0.16, 0.2]}
        precision={4}
        getValue={(t) => t.motionEnergy}
        telemetryRef={telemetryRef}
      />

      {/* Bottom Left: PID Gain Controller */}
      <PIDNode
        gains={pidGains}
        onUpdate={onUpdatePidGains}
        onReset={onResetPidGains}
        disabled={pidGainsLoading}
      />

      {/* Bottom Center: Micro Telemetry Readout Strip */}
      <TelemetryStrip telemetryRef={telemetryRef} />

      {/* Bottom Right: Mode Controls and Actions */}
      <ModeNode
        runtimeConfig={runtimeConfig}
        runtimeConfigLoading={runtimeConfigLoading}
        running={running}
        modelLoaded={modelLoaded}
        cameraIndex={cameraIndex}
        onLoadModel={onLoadModel}
        onStartSimulation={onStartSimulation}
        onStopSimulation={onStopSimulation}
        onSetConfig={onSetRuntimeConfig}
        onSetCameraSource={onSetCameraSource}
      />
    </div>
  );
}
