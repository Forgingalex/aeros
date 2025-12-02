import React, { useState, useEffect, useCallback } from "react";
import "./PIDControlPanel.css";

// PID Presets for different scenarios
const PID_PRESETS = {
  default: { kp: 1.0, ki: 0.1, kd: 0.5, label: "Default" },
  aggressive: { kp: 2.0, ki: 0.2, kd: 1.0, label: "Aggressive" },
  smooth: { kp: 0.5, ki: 0.05, kd: 0.3, label: "Smooth" },
  fast: { kp: 1.5, ki: 0.15, kd: 0.8, label: "Fast Response" },
  stable: { kp: 0.8, ki: 0.2, kd: 0.6, label: "Stable" },
};

const PIDControlPanel = ({
  currentGains,
  onUpdateGains,
  onReset,
  disabled = false,
}) => {
  const [localGains, setLocalGains] = useState({
    kp: currentGains?.kp || 1.0,
    ki: currentGains?.ki || 0.1,
    kd: currentGains?.kd || 0.5,
  });
  const [isUpdating, setIsUpdating] = useState(false);

  // Sync with prop changes
  useEffect(() => {
    if (currentGains) {
      setLocalGains({
        kp: currentGains.kp,
        ki: currentGains.ki,
        kd: currentGains.kd,
      });
    }
  }, [currentGains]);

  const handleGainChange = useCallback(
    (gain, value) => {
      const numValue = parseFloat(value) || 0;
      const newGains = { ...localGains, [gain]: numValue };
      setLocalGains(newGains);
    },
    [localGains]
  );

  const handleApply = useCallback(async () => {
    setIsUpdating(true);
    try {
      await onUpdateGains(localGains);
    } finally {
      setIsUpdating(false);
    }
  }, [localGains, onUpdateGains]);

  const handlePreset = useCallback(
    async (preset) => {
      setLocalGains(preset);
      setIsUpdating(true);
      try {
        await onUpdateGains(preset);
      } finally {
        setIsUpdating(false);
      }
    },
    [onUpdateGains]
  );

  const handleReset = useCallback(async () => {
    const defaultPreset = PID_PRESETS.default;
    setLocalGains(defaultPreset);
    setIsUpdating(true);
    try {
      await onReset();
    } finally {
      setIsUpdating(false);
    }
  }, [onReset]);

  return (
    <div className="aeros-card aeros-panel aeros-pid-panel">
      <div className="aeros-panel-header-row">
        <h2 className="aeros-panel-title">PID Controller</h2>
        <button
          className="aeros-btn aeros-btn-ghost aeros-btn-sm"
          onClick={handleReset}
          disabled={disabled || isUpdating}
          title="Reset to default values"
        >
          Reset
        </button>
      </div>

      {/* Current Values Display */}
      <div className="aeros-pid-current">
        <div className="aeros-pid-current-label">Current Values</div>
        <div className="aeros-pid-current-values">
          <span>Kp: {currentGains?.kp?.toFixed(3) || "—"}</span>
          <span>Ki: {currentGains?.ki?.toFixed(3) || "—"}</span>
          <span>Kd: {currentGains?.kd?.toFixed(3) || "—"}</span>
        </div>
      </div>

      {/* Presets */}
      <div className="aeros-pid-presets">
        <div className="aeros-pid-presets-label">Presets</div>
        <div className="aeros-pid-presets-grid">
          {Object.entries(PID_PRESETS).map(([key, preset]) => (
            <button
              key={key}
              className="aeros-btn aeros-btn-outline aeros-btn-sm"
              onClick={() => handlePreset(preset)}
              disabled={disabled || isUpdating}
            >
              {preset.label}
            </button>
          ))}
        </div>
      </div>

      {/* Gain Controls */}
      <div className="aeros-pid-controls">
        {["kp", "ki", "kd"].map((gain) => (
          <div key={gain} className="aeros-pid-control">
            <div className="aeros-pid-control-header">
              <label className="aeros-pid-control-label">
                {gain.toUpperCase()}
              </label>
              <input
                type="number"
                className="aeros-pid-input"
                value={localGains[gain]}
                onChange={(e) => handleGainChange(gain, e.target.value)}
                step={gain === "ki" ? 0.01 : 0.1}
                min="0"
                max={gain === "kp" ? "5" : gain === "ki" ? "1" : "3"}
                disabled={disabled || isUpdating}
              />
            </div>
            <input
              type="range"
              className="aeros-pid-slider"
              value={localGains[gain]}
              onChange={(e) => handleGainChange(gain, e.target.value)}
              step={gain === "ki" ? 0.01 : 0.1}
              min="0"
              max={gain === "kp" ? "5" : gain === "ki" ? "1" : "3"}
              disabled={disabled || isUpdating}
            />
          </div>
        ))}
      </div>

      {/* Apply Button */}
      <div className="aeros-pid-apply">
        <button
          className="aeros-btn aeros-btn-primary"
          onClick={handleApply}
          disabled={
            disabled ||
            isUpdating ||
            (currentGains &&
              localGains.kp === currentGains.kp &&
              localGains.ki === currentGains.ki &&
              localGains.kd === currentGains.kd)
          }
        >
          {isUpdating ? "Applying..." : "Apply Changes"}
        </button>
      </div>
    </div>
  );
};

export default PIDControlPanel;

