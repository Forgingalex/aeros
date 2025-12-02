import React from "react";
import "./AerosDashboard.css";
import Logo from "./Logo";
import PIDControlPanel from "./PIDControlPanel";

const StatusBadge = ({ status }) => {
  const label =
    status === "connected"
      ? "Connected"
      : status === "connecting"
      ? "Connecting"
      : "Disconnected";

  return (
    <div className={`aeros-status aeros-status-${status}`}>
      <span className="aeros-status-dot" />
      <span className="aeros-status-label">{label}</span>
    </div>
  );
};

const MetricCard = ({
  label,
  value,
  unit,
  accent = "neutral",
}) => {
  return (
    <div className={`aeros-card aeros-metric aeros-metric-${accent}`}>
      <div className="aeros-metric-label">{label}</div>
      <div className="aeros-metric-value-row">
        <span className="aeros-metric-value">{value}</span>
        {unit && <span className="aeros-metric-unit">{unit}</span>}
      </div>
    </div>
  );
};

const CameraPanel = ({
  isConnected,
  cameraUrl,
}) => {
  return (
    <div className="aeros-card aeros-panel">
      <div className="aeros-panel-header-row">
        <h2 className="aeros-panel-title">Camera Feed</h2>
      </div>
      <div className="aeros-camera-frame">
        {isConnected && cameraUrl ? (
          <img
            src={cameraUrl}
            alt="Camera feed"
            className="aeros-camera-img"
          />
        ) : (
          <div className="aeros-camera-placeholder">
            <div className="aeros-camera-icon" />
            <div className="aeros-camera-text-main">Waiting for connection</div>
            <div className="aeros-camera-text-sub">
              Ensure API server is running
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

const TelemetryPanel = ({ headingDeg }) => {
  return (
    <div className="aeros-card aeros-panel">
      <div className="aeros-panel-header-row">
        <h2 className="aeros-panel-title">Telemetry</h2>
      </div>
      <div className="aeros-telemetry">
        <div className="aeros-compass">
          <div className="aeros-compass-inner">
            <div
              className="aeros-compass-needle"
              style={{ transform: `rotate(${headingDeg}deg)` }}
            />
            <div className="aeros-compass-center" />
            <div className="aeros-compass-label aeros-compass-n">N</div>
            <div className="aeros-compass-label aeros-compass-e">E</div>
            <div className="aeros-compass-label aeros-compass-s">S</div>
            <div className="aeros-compass-label aeros-compass-w">W</div>
          </div>
        </div>
        <div className="aeros-heading-pill">{headingDeg.toFixed(1)}°</div>
      </div>
    </div>
  );
};

const ControlPanel = ({
  onLoadModel,
  onStart,
  onStop,
  modelLoaded,
  running,
}) => {
  return (
    <div className="aeros-card aeros-panel">
      <div className="aeros-panel-header-row">
        <h2 className="aeros-panel-title">Control</h2>
      </div>
      <div className="aeros-control-row">
        <button
          className="aeros-btn aeros-btn-outline"
          onClick={onLoadModel}
          disabled={modelLoaded}
        >
          {modelLoaded ? "Model loaded" : "Load model"}
        </button>
        <button
          className="aeros-btn aeros-btn-primary"
          onClick={onStart}
          disabled={!modelLoaded || running}
        >
          Start
        </button>
        <button
          className="aeros-btn aeros-btn-ghost"
          onClick={onStop}
          disabled={!running}
        >
          Stop
        </button>
      </div>
      {!modelLoaded && (
        <div className="aeros-inline-alert">
          Model not loaded. Call /load_model first.
        </div>
      )}
    </div>
  );
};

/**
 * Main dashboard layout
 */
export const AerosDashboard = (props) => {
  const {
    status,
    cameraUrl,
    headingDeg,
    controlOutput,
    fps,
    latencyMs,
    framesProcessed,
    onLoadModel,
    onStart,
    onStop,
    modelLoaded,
    running,
    pidGains,
    onUpdatePIDGains,
    onResetPIDGains,
    pidGainsLoading,
  } = props;

  return (
    <div className="aeros-root">
      <header className="aeros-header">
        <div className="aeros-header-left">
          <div className="aeros-logo-wrap">
            <Logo size={40} />
          </div>
          <div>
            <div className="aeros-wordmark">AEROS</div>
            <div className="aeros-tagline">Autonomy Pipeline</div>
          </div>
        </div>
        <div className="aeros-header-right">
          <StatusBadge status={status} />
        </div>
      </header>

      <main className="aeros-main">
        <section className="aeros-main-left">
          <CameraPanel
            isConnected={status === "connected"}
            cameraUrl={cameraUrl}
          />
        </section>

        <section className="aeros-main-right">
          <TelemetryPanel headingDeg={headingDeg} />

          <ControlPanel
            onLoadModel={onLoadModel}
            onStart={onStart}
            onStop={onStop}
            modelLoaded={modelLoaded}
            running={running}
          />

          <PIDControlPanel
            currentGains={pidGains}
            onUpdateGains={onUpdatePIDGains}
            onReset={onResetPIDGains}
            disabled={!modelLoaded || pidGainsLoading}
          />

          <div className="aeros-metrics-grid">
            <MetricCard
              label="Control output"
              value={controlOutput.toFixed(4)}
              unit="rad/s"
              accent="brand"
            />
            <MetricCard
              label="FPS"
              value={fps.toFixed(1)}
              accent={fps < 10 ? "warning" : "success"}
            />
            <MetricCard
              label="Latency"
              value={latencyMs.toFixed(2)}
              unit="ms"
              accent={latencyMs > 200 ? "warning" : "success"}
            />
            <MetricCard
              label="Frames processed"
              value={framesProcessed}
              accent="neutral"
            />
          </div>
        </section>
      </main>
    </div>
  );
};

