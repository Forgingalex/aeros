import React from 'react';
import './Logo.css';

/**
 * AEROS Logo Component
 * Abstract geometric flight vector icon - two intertwined elements:
 * 1. Stylized arrow pointing diagonally upwards and to the right
 * 2. Elongated open oval that the arrow passes through
 */
const Logo = ({ size = 48 }) => {
  return (
    <div className="aeros-logo" style={{ width: size, height: size }}>
      <svg
        width={size}
        height={size}
        viewBox="0 0 100 100"
        className="logo-svg"
        xmlns="http://www.w3.org/2000/svg"
        fill="none"
      >
        {/* Elongated open oval - passes through the arrow */}
        <ellipse
          cx="50"
          cy="50"
          rx="35"
          ry="20"
          stroke="#1C7DF2"
          strokeWidth="3"
          fill="none"
          className="logo-oval"
        />
        
        {/* Stylized arrow - curved shaft with triangular arrowhead */}
        <g className="logo-arrow">
          {/* Curved arrow shaft */}
          <path
            d="M 15 75 Q 30 60, 50 50 Q 70 40, 85 25"
            stroke="#1C7DF2"
            strokeWidth="4"
            strokeLinecap="round"
            fill="none"
            className="logo-arrow-shaft"
          />
          
          {/* Arrowhead */}
          <path
            d="M 75 30 L 85 25 L 80 20 Z"
            fill="#1C7DF2"
            className="logo-arrowhead"
          />
        </g>
      </svg>
    </div>
  );
};

export default Logo;
