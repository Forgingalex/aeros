import React, { useEffect, useRef } from "react";

// Scale factor: how many pixels the velocity vector shifts per unit of heading_rate
const VELOCITY_VECTOR_SCALE = 60;

// Pitch ladder marks in degrees
const PITCH_MARKS = [10, 20, 30];

export default function BoreSight({ telemetryRef }) {
  const velocityVectorRef = useRef(null);
  const aoaLeftRef = useRef(null);
  const aoaRightRef = useRef(null);
  const requestRef = useRef(null);

  useEffect(() => {
    const update = () => {
      const t = telemetryRef.current;
      if (!t) {
        requestRef.current = requestAnimationFrame(update);
        return;
      }

      // Velocity Vector: shift horizontally based on heading_rate
      if (velocityVectorRef.current) {
        const headingRate = t.headingRate || 0;
        const offsetX = Math.max(-40, Math.min(40, headingRate * VELOCITY_VECTOR_SCALE));
        velocityVectorRef.current.style.transform = `translateX(${offsetX}px)`;
      }

      // AOA Brackets: pulse opacity based on motion_energy
      const energy = t.motionEnergy != null ? t.motionEnergy : 0;
      const aoaOpacity = Math.min(1.0, 0.3 + (energy / 0.2) * 0.7);
      const aoaScale = 1.0 + energy * 0.5;

      if (aoaLeftRef.current) {
        aoaLeftRef.current.style.opacity = aoaOpacity;
        aoaLeftRef.current.style.transform = `scaleY(${aoaScale})`;
      }
      if (aoaRightRef.current) {
        aoaRightRef.current.style.opacity = aoaOpacity;
        aoaRightRef.current.style.transform = `scaleY(${aoaScale})`;
      }

      requestRef.current = requestAnimationFrame(update);
    };

    requestRef.current = requestAnimationFrame(update);
    return () => {
      if (requestRef.current) cancelAnimationFrame(requestRef.current);
    };
  }, [telemetryRef]);

  return (
    <div className="hud-boresight">
      <svg
        className="hud-boresight-svg"
        viewBox="0 0 200 200"
        xmlns="http://www.w3.org/2000/svg"
      >
        {/* ── Pitch Ladder ── */}
        {PITCH_MARKS.map((deg) => {
          const yUp = 100 - deg * 1.8;
          const yDown = 100 + deg * 1.8;
          return (
            <g key={deg}>
              {/* Upper bar */}
              <line x1="60" y1={yUp} x2="90" y2={yUp} className="hud-boresight-pitch-bar" />
              <line x1="110" y1={yUp} x2="140" y2={yUp} className="hud-boresight-pitch-bar" />
              <text x="55" y={yUp + 3} className="hud-boresight-pitch-label" textAnchor="end">{deg}</text>
              <text x="145" y={yUp + 3} className="hud-boresight-pitch-label" textAnchor="start">{deg}</text>
              {/* Lower bar */}
              <line x1="65" y1={yDown} x2="90" y2={yDown} className="hud-boresight-pitch-bar lower" />
              <line x1="110" y1={yDown} x2="135" y2={yDown} className="hud-boresight-pitch-bar lower" />
              <text x="60" y={yDown + 3} className="hud-boresight-pitch-label" textAnchor="end">-{deg}</text>
              <text x="140" y={yDown + 3} className="hud-boresight-pitch-label" textAnchor="start">-{deg}</text>
            </g>
          );
        })}

        {/* ── Static Bore-Sight Frame ── */}
        {/* Left wing */}
        <line x1="24" y1="100" x2="72" y2="100" className="hud-boresight-line hud-boresight-glow" />
        <line x1="24" y1="100" x2="24" y2="90" className="hud-boresight-line hud-boresight-glow" />
        {/* Right wing */}
        <line x1="128" y1="100" x2="176" y2="100" className="hud-boresight-line hud-boresight-glow" />
        <line x1="176" y1="100" x2="176" y2="90" className="hud-boresight-line hud-boresight-glow" />
        {/* Top tick */}
        <line x1="100" y1="30" x2="100" y2="62" className="hud-boresight-line hud-boresight-glow" />
        {/* Bottom tick */}
        <line x1="100" y1="138" x2="100" y2="170" className="hud-boresight-line hud-boresight-glow" />

        {/* ── Outer dotted ring ── */}
        <circle cx="100" cy="100" r="60" className="hud-boresight-circle hud-boresight-glow" />

        {/* ── Center dot ── */}
        <circle cx="100" cy="100" r="3" className="hud-boresight-center-dot hud-boresight-glow" />
      </svg>

      {/* ── Velocity Vector (Ghost Icon) ── */}
      <div ref={velocityVectorRef} className="hud-boresight-velocity-vector">
        <svg width="28" height="20" viewBox="0 0 28 20" xmlns="http://www.w3.org/2000/svg">
          <circle cx="14" cy="10" r="4" fill="none" stroke="var(--hud-cyan)" strokeWidth="1.2" className="hud-boresight-glow" />
          {/* Left wing */}
          <line x1="0" y1="10" x2="10" y2="10" stroke="var(--hud-cyan)" strokeWidth="1.2" className="hud-boresight-glow" />
          {/* Right wing */}
          <line x1="18" y1="10" x2="28" y2="10" stroke="var(--hud-cyan)" strokeWidth="1.2" className="hud-boresight-glow" />
          {/* Top tail */}
          <line x1="14" y1="6" x2="14" y2="0" stroke="var(--hud-cyan)" strokeWidth="1.2" className="hud-boresight-glow" />
        </svg>
      </div>

      {/* ── AOA Brackets ── */}
      <div ref={aoaLeftRef} className="hud-boresight-aoa-bracket left">[</div>
      <div ref={aoaRightRef} className="hud-boresight-aoa-bracket right">]</div>
    </div>
  );
}
