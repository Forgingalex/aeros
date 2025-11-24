import React from 'react';
import './Logo.css';

const Logo = ({ size = 48 }) => {
  return (
    <div className="logo-container">
      <svg
        width={size}
        height={size}
        viewBox="0 0 100 100"
        className="logo-svg"
        xmlns="http://www.w3.org/2000/svg"
      >
        {/* Drone body */}
        <ellipse
          cx="50"
          cy="50"
          rx="25"
          ry="15"
          fill="#ba1c34"
          className="logo-drone-body"
        />
        {/* Propellers */}
        <circle cx="30" cy="40" r="8" fill="#ffffff" opacity="0.9" className="logo-propeller" />
        <circle cx="70" cy="40" r="8" fill="#ffffff" opacity="0.9" className="logo-propeller" />
        <circle cx="30" cy="60" r="8" fill="#ffffff" opacity="0.9" className="logo-propeller" />
        <circle cx="70" cy="60" r="8" fill="#ffffff" opacity="0.9" className="logo-propeller" />
        {/* Center dot */}
        <circle cx="50" cy="50" r="4" fill="#ffffff" />
        {/* Motion lines */}
        <path
          d="M 20 50 L 30 50"
          stroke="#ba1c34"
          strokeWidth="2"
          strokeLinecap="round"
          className="logo-motion"
        />
        <path
          d="M 70 50 L 80 50"
          stroke="#ba1c34"
          strokeWidth="2"
          strokeLinecap="round"
          className="logo-motion"
        />
      </svg>
      <span className="logo-text">AEROS</span>
    </div>
  );
};

export default Logo;

