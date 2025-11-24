import React from 'react';
import './HeadingIndicator.css';

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
          stroke="rgba(186, 28, 52, 0.3)"
          strokeWidth="2"
        />
        
        {/* Degree markers */}
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
              stroke="rgba(255, 255, 255, 0.5)"
              strokeWidth="2"
            />
          );
        })}
        
        {/* Cardinal directions */}
        <text x="100" y="25" textAnchor="middle" fill="#ba1c34" fontSize="16" fontWeight="700">N</text>
        <text x="100" y="185" textAnchor="middle" fill="rgba(255, 255, 255, 0.6)" fontSize="14">S</text>
        <text x="25" y="105" textAnchor="middle" fill="rgba(255, 255, 255, 0.6)" fontSize="14">W</text>
        <text x="175" y="105" textAnchor="middle" fill="rgba(255, 255, 255, 0.6)" fontSize="14">E</text>
        
        {/* Heading arrow */}
        <g transform={`rotate(${angle} 100 100)`}>
          <path
            d="M 100 20 L 95 50 L 100 45 L 105 50 Z"
            fill="#ba1c34"
            className="heading-arrow"
          />
          <line
            x1="100"
            y1="20"
            x2="100"
            y2="80"
            stroke="#ba1c34"
            strokeWidth="3"
            strokeLinecap="round"
          />
        </g>
        
        {/* Center dot */}
        <circle cx="100" cy="100" r="6" fill="#ba1c34" />
        <circle cx="100" cy="100" r="3" fill="#ffffff" />
      </svg>
      
      <div className="heading-value-display">
        {angle.toFixed(1)}°
      </div>
    </div>
  );
};

export default HeadingIndicator;

