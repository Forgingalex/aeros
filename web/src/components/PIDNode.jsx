import React, { useEffect, useState, useRef } from "react";

const PID_PRESETS = {
  nominal: { kp: 1.0, ki: 0.1, kd: 0.5, label: "Nominal" },
  precision: { kp: 0.75, ki: 0.08, kd: 0.4, label: "Precision" },
  aggressive: { kp: 2.0, ki: 0.18, kd: 1.0, label: "Aggressive" },
  damped: { kp: 0.6, ki: 0.16, kd: 0.75, label: "Damped" },
};

const GAIN_META = {
  kp: {
    label: "Kp",
    longLabel: "Proportional",
    min: 0,
    max: 5,
    step: 0.1,
  },
  ki: {
    label: "Ki",
    longLabel: "Integral",
    min: 0,
    max: 1,
    step: 0.01,
  },
  kd: {
    label: "Kd",
    longLabel: "Derivative",
    min: 0,
    max: 3,
    step: 0.1,
  },
};

function presetToGains(preset) {
  return {
    kp: preset.kp,
    ki: preset.ki,
    kd: preset.kd,
  };
}

export default function PIDNode({
  gains,
  onUpdate,
  onReset,
  disabled = false,
}) {
  const [localGains, setLocalGains] = useState({
    kp: gains?.kp ?? 1.0,
    ki: gains?.ki ?? 0.1,
    kd: gains?.kd ?? 0.5,
  });

  // Track the timer for debouncing API calls
  const debounceTimerRef = useRef(null);

  // Sync with remote gains when they change externally
  useEffect(() => {
    if (gains) {
      setLocalGains({
        kp: gains.kp,
        ki: gains.ki,
        kd: gains.kd,
      });
    }
  }, [gains]);

  // Clean up timer on unmount
  useEffect(() => {
    return () => {
      if (debounceTimerRef.current) {
        clearTimeout(debounceTimerRef.current);
      }
    };
  }, []);

  const triggerUpdate = (nextGains) => {
    if (debounceTimerRef.current) {
      clearTimeout(debounceTimerRef.current);
    }
    
    // Debounce the actual API call by 150ms so dragging is smooth and doesn't spam the server
    debounceTimerRef.current = setTimeout(() => {
      onUpdate(nextGains);
    }, 150);
  };

  const handleSliderChange = (key, nextValue) => {
    const val = parseFloat(nextValue);
    if (isNaN(val)) return;

    const nextGains = {
      ...localGains,
      [key]: val,
    };
    setLocalGains(nextGains);
    triggerUpdate(nextGains);
  };

  const handlePreset = (preset) => {
    const nextGains = presetToGains(preset);
    setLocalGains(nextGains);
    onUpdate(nextGains);
  };

  const handleReset = () => {
    setLocalGains(presetToGains(PID_PRESETS.nominal));
    onReset();
  };

  return (
    <div className="hud-node hud-pid-node">
      <div className="hud-node-title-row">
        <span className="hud-node-title">PID Gains</span>
        <button
          type="button"
          className="hud-node-reset-btn"
          onClick={handleReset}
          disabled={disabled}
        >
          Reset
        </button>
      </div>

      {/* Preset Badges (Segmented HUD style) */}
      <div className="hud-control-block">
        <div className="hud-segmented">
          {Object.entries(PID_PRESETS).map(([key, preset]) => {
            const isMatch =
              Math.abs(localGains.kp - preset.kp) < 0.01 &&
              Math.abs(localGains.ki - preset.ki) < 0.01 &&
              Math.abs(localGains.kd - preset.kd) < 0.01;
            return (
              <button
                key={key}
                type="button"
                className={`hud-segment-button ${isMatch ? "is-active" : ""}`}
                disabled={disabled}
                onClick={() => handlePreset(preset)}
              >
                {preset.label}
              </button>
            );
          })}
        </div>
      </div>

      {/* Custom premium slider rails */}
      {Object.entries(GAIN_META).map(([key, meta]) => {
        const val = localGains[key] ?? 0;
        const pct = ((val - meta.min) / (meta.max - meta.min)) * 100;
        
        return (
          <div key={key} className="hud-slider-group">
            <div className="hud-slider-header">
              <span className="hud-slider-name">
                {meta.label} <span style={{ fontSize: "9px", color: "var(--hud-text-dim)", fontWeight: "normal" }}>({meta.longLabel})</span>
              </span>
              <span className="hud-slider-value">
                {val.toFixed(key === "ki" ? 2 : 1)}
              </span>
            </div>
            
            <div className="hud-slider-wrapper">
              <input
                type="range"
                className="hud-slider-input"
                min={meta.min}
                max={meta.max}
                step={meta.step}
                value={val}
                disabled={disabled}
                style={{
                  background: `linear-gradient(90deg, var(--hud-electric) 0%, var(--hud-electric) ${pct}%, rgba(28, 125, 242, 0.1) ${pct}%, rgba(28, 125, 242, 0.1) 100%)`,
                }}
                onChange={(e) => handleSliderChange(key, e.target.value)}
              />
            </div>
            
            <div className="hud-slider-ticks">
              <span>{meta.min}</span>
              <span>{((meta.max + meta.min) / 2).toFixed(1)}</span>
              <span>{meta.max}</span>
            </div>
          </div>
        );
      })}
    </div>
  );
}
