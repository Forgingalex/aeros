import React, { useEffect, useRef } from "react";

const DEGREE_PX = 4.4;
const VIEWPORT_WIDTH = 340;
const CENTER_X = VIEWPORT_WIDTH / 2;

function normalizeHeading(value) {
  const normalized = value % 360;
  return normalized < 0 ? normalized + 360 : normalized;
}

export default function HeadingTape({ telemetryRef }) {
  const trackRef = useRef(null);
  const readoutRef = useRef(null);
  const requestRef = useRef(null);

  const ticks = [];
  for (let t = -18; t <= 54; t++) {
    const heading = (t * 10 + 360) % 360;
    let displayLabel = String(heading).padStart(3, "0");
    let isCardinal = false;

    if (heading === 0)   { displayLabel = "N"; isCardinal = true; }
    else if (heading === 90)  { displayLabel = "E"; isCardinal = true; }
    else if (heading === 180) { displayLabel = "S"; isCardinal = true; }
    else if (heading === 270) { displayLabel = "W"; isCardinal = true; }

    const degFromStart = t * 10 - (-180);
    const leftPx = degFromStart * DEGREE_PX;

    ticks.push({ key: t, leftPx, displayLabel, isCardinal });
  }

  useEffect(() => {
    const updateHeading = () => {
      const telemetry = telemetryRef.current;
      if (!telemetry) {
        requestRef.current = requestAnimationFrame(updateHeading);
        return;
      }

      const rawHeading = telemetry.headingDegrees || 0;
      const heading = normalizeHeading(rawHeading);

      if (trackRef.current) {
        const headingFromStart = heading - (-180);
        const tx = CENTER_X - (headingFromStart * DEGREE_PX);
        trackRef.current.style.transform = `translateX(${tx}px)`;
      }

      if (readoutRef.current) {
        readoutRef.current.textContent = `HDG ${String(Math.round(heading)).padStart(3, "0")}°`;
      }

      requestRef.current = requestAnimationFrame(updateHeading);
    };

    requestRef.current = requestAnimationFrame(updateHeading);
    return () => {
      if (requestRef.current) cancelAnimationFrame(requestRef.current);
    };
  }, [telemetryRef]);

  return (
    <div className="hud-heading-tape-container">
      <div className="hud-heading-tape-caret" />

      <div className="hud-heading-tape-window">
        {/* Sliding compass track */}
        <div ref={trackRef} className="hud-heading-tape-track">
          {ticks.map((tick) => (
            <div
              key={tick.key}
              className="hud-heading-tape-tick"
              style={{ left: `${tick.leftPx}px` }}
            >
              <span className={`hud-heading-tape-tick-mark ${tick.isCardinal ? "major" : ""}`} />
              <span className={`hud-heading-tape-tick-label ${tick.isCardinal ? "major" : ""}`}>
                {tick.displayLabel}
              </span>
            </div>
          ))}
        </div>

        {/* ── Stationary Readout Box (Aviation Window) ── */}
        <div className="hud-heading-readout-box">
          <span ref={readoutRef} className="hud-heading-readout-value">HDG 000°</span>
        </div>
      </div>
    </div>
  );
}
