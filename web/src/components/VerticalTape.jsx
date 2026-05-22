import React, { useEffect, useRef } from "react";

const VIEWPORT_HEIGHT = 160;
const CENTER_Y = VIEWPORT_HEIGHT / 2;
const TRACK_HEIGHT = 320;

export default function VerticalTape({
  side,
  label,
  min,
  max,
  marks,
  precision = 1,
  getValue,
  telemetryRef,
}) {
  const containerRef = useRef(null);
  const trackRef = useRef(null);
  const readoutValRef = useRef(null);
  const requestRef = useRef(null);

  const pixelsPerUnit = TRACK_HEIGHT / (max - min);

  useEffect(() => {
    const updateTape = () => {
      const telemetry = telemetryRef.current;
      if (!telemetry) {
        requestRef.current = requestAnimationFrame(updateTape);
        return;
      }

      const val = getValue(telemetry);
      const isDormant = val == null || val <= 0;

      if (containerRef.current) {
        containerRef.current.classList.toggle("dormant", isDormant);
      }

      if (readoutValRef.current) {
        if (isDormant) {
          readoutValRef.current.textContent = "---";
        } else {
          readoutValRef.current.textContent = val.toFixed(precision);
        }
      }

      if (trackRef.current) {
        const activeVal = isDormant ? 0 : Math.max(min, Math.min(max, val));
        const tickTop = (max - activeVal) * pixelsPerUnit;
        const translateY = CENTER_Y - tickTop;
        trackRef.current.style.transform = `translateY(${translateY}px)`;
      }

      requestRef.current = requestAnimationFrame(updateTape);
    };

    requestRef.current = requestAnimationFrame(updateTape);
    return () => {
      if (requestRef.current) cancelAnimationFrame(requestRef.current);
    };
  }, [telemetryRef, min, max, pixelsPerUnit, getValue, precision]);

  return (
    <div
      ref={containerRef}
      className={`hud-vertical-tape-container ${side} dormant`}
    >
      <div className="hud-vertical-tape-label">{label}</div>

      <div className="hud-vertical-tape-viewport">
        {/* Fixed Center Pointer Caret */}
        <div className="hud-vertical-tape-pointer" />

        {/* Sliding Tick Track */}
        <div ref={trackRef} className="hud-vertical-tape-track">
          {marks.map((mark) => {
            const topPx = (max - mark) * pixelsPerUnit;
            const isMajor = mark % 10 === 0 || mark === max || mark === min || (max - min <= 1);
            return (
              <div
                key={mark}
                className="hud-vertical-tape-tick"
                style={{ top: `${topPx}px` }}
              >
                {side === "left" && (
                  <span className={`hud-vertical-tape-tick-label ${isMajor ? "major" : ""}`}>
                    {mark.toFixed(precision === 0 ? 0 : 1)}
                  </span>
                )}
                <span className={`hud-vertical-tape-tick-line ${isMajor ? "major" : ""}`} />
                {side === "right" && (
                  <span className={`hud-vertical-tape-tick-label ${isMajor ? "major" : ""}`}>
                    {mark.toFixed(precision === 0 ? 0 : 1)}
                  </span>
                )}
              </div>
            );
          })}
        </div>

        {/* ── Stationary Readout Box (Aviation Window) ── */}
        <div className={`hud-vertical-tape-readout-box ${side}`}>
          <span ref={readoutValRef} className="hud-vertical-tape-readout-value">---</span>
        </div>
      </div>
    </div>
  );
}
