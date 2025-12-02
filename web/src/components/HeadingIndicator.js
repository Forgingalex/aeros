import React from 'react';
import './HeadingIndicator.css';

/**
 * Heading Indicator Component
 * Aerospace precision aesthetic with crisp vector motion indicator
 */
const HeadingIndicator = ({ heading, size = 200 }) => {
  const angle = (heading * 180) / Math.PI;
  
  return (
    <div className="heading-indicator-container" style={{ width: size, height: size }}>
      <svg
        width={size}
        height={size}
        viewBox="0 0 200 200"
        className="heading-compass"
      >
        {/* Outer ring */}
        <circle
          cx="100"
          cy="100"
          r="90"
          fill="none"
          stroke="rgba(255, 255, 255, 0.1)"
          strokeWidth="1.5"
        />
        
        {/* Degree markers - line-based, precise */}
        {[0, 45, 90, 135, 180, 225, 270, 315].map((deg) => {
          const rad = (deg * Math.PI) / 180;
          const x1 = 100 + 85 * Math.cos(rad);
          const y1 = 100 + 85 * Math.sin(rad);
          const x2 = 100 + 95 * Math.cos(rad);
          const y2 = 100 + 95 * Math.sin(rad);
          return (
            <line
              key={deg}
              x1={x1}
              y1={y1}
              x2={x2}
              y2={y2}
              stroke="rgba(255, 255, 255, 0.3)"
              strokeWidth="1.5"
              strokeLinecap="round"
            />
          );
        })}
        
        {/* Cardinal directions - minimal, line-based */}
        <text x="100" y="25" textAnchor="middle" fill="#1C7DF2" fontSize="14" fontWeight="600">N</text>
        <text x="100" y="185" textAnchor="middle" fill="rgba(255, 255, 255, 0.5)" fontSize="12" fontWeight="400">S</text>
        <text x="25" y="105" textAnchor="middle" fill="rgba(255, 255, 255, 0.5)" fontSize="12" fontWeight="400">W</text>
        <text x="175" y="105" textAnchor="middle" fill="rgba(255, 255, 255, 0.5)" fontSize="12" fontWeight="400">E</text>
        
        {/* Heading arrow - crisp vector motion indicator */}
        <g transform={`rotate(${angle} 100 100)`}>
          {/* Arrow shaft - line-based */}
          <line
            x1="100"
            y1="20"
            x2="100"
            y2="80"
            stroke="#1C7DF2"
            strokeWidth="3"
            strokeLinecap="round"
            className="heading-arrow"
          />
          {/* Arrowhead - triangular, precise */}
          <path
            d="M 100 20 L 95 35 L 100 30 L 105 35 Z"
            fill="#1C7DF2"
            className="heading-arrowhead"
          />
        </g>
        
        {/* Center dot - minimal */}
        <circle cx="100" cy="100" r="4" fill="#1C7DF2" />
        <circle cx="100" cy="100" r="2" fill="rgba(255, 255, 255, 0.8)" />
      </svg>
      
      <div className="heading-value-display">
        {angle.toFixed(1)}°
      </div>
    </div>
  );
};

export default HeadingIndicator;
