import React from "react";

export default function ModeNode({
  runtimeConfig,
  runtimeConfigLoading,
  running,
  modelLoaded,
  cameraIndex = 0,
  onLoadModel,
  onStartSimulation,
  onStopSimulation,
  onSetConfig,
  onSetCameraSource,
}) {
  return (
    <div className="hud-node hud-mode-node">
      <div className="hud-node-title-row">
        <span className="hud-node-title">Command Surface</span>
      </div>

      {/* Runtime Mode Segmented Control */}
      <div className="hud-control-block">
        <span className="hud-control-label">Runtime Mode</span>
        <div className="hud-segmented">
          <button
            type="button"
            className={`hud-segment-button ${
              runtimeConfig.runtime_mode === "legacy" ? "is-active" : ""
            }`}
            disabled={runtimeConfigLoading}
            onClick={() => onSetConfig({ runtime_mode: "legacy" })}
          >
            Legacy
          </button>
          <button
            type="button"
            className={`hud-segment-button ${
              runtimeConfig.runtime_mode === "omega_shadow" ? "is-active" : ""
            }`}
            disabled={runtimeConfigLoading}
            onClick={() => onSetConfig({ runtime_mode: "omega_shadow" })}
          >
            Shadow
          </button>
        </div>
      </div>

      {/* Stream View Segmented Control */}
      <div className="hud-control-block">
        <span className="hud-control-label">Stream View</span>
        <div className="hud-segmented">
          <button
            type="button"
            className={`hud-segment-button ${
              runtimeConfig.stream_view === "auto" ? "is-active" : ""
            }`}
            disabled={runtimeConfigLoading}
            onClick={() => onSetConfig({ stream_view: "auto" })}
          >
            Auto
          </button>
          <button
            type="button"
            className={`hud-segment-button ${
              runtimeConfig.stream_view === "rgb" ? "is-active" : ""
            }`}
            disabled={runtimeConfigLoading}
            onClick={() => onSetConfig({ stream_view: "rgb" })}
          >
            RGB
          </button>
          <button
            type="button"
            className={`hud-segment-button ${
              runtimeConfig.stream_view === "delta" ? "is-active" : ""
            }`}
            disabled={runtimeConfigLoading}
            onClick={() => onSetConfig({ stream_view: "delta" })}
          >
            Delta
          </button>
        </div>
      </div>

      {/* Action Buttons Row */}
      <div className="hud-action-row">
        <button
          type="button"
          className={`hud-btn ${
            modelLoaded ? "hud-btn-ghost" : "hud-btn-primary"
          }`}
          disabled={modelLoaded}
          onClick={onLoadModel}
        >
          {modelLoaded ? "LOCKED" : "LOAD NET"}
        </button>

        {!running ? (
          <button
            type="button"
            className="hud-btn hud-btn-outline"
            onClick={onStartSimulation}
          >
            TRY SIM
          </button>
        ) : (
          <button
            type="button"
            className="hud-btn hud-btn-primary"
            onClick={onStopSimulation}
          >
            CAM FEED
          </button>
        )}

        <button
          type="button"
          className="hud-btn hud-btn-outline hud-cam-toggle"
          onClick={() => {
            const nextIndex = cameraIndex === 0 ? 1 : 0;
            if (onSetCameraSource) onSetCameraSource(nextIndex);
          }}
        >
          CAM {cameraIndex}
        </button>
      </div>
    </div>
  );
}
