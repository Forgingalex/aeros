import React from 'react';
import './MetricCard.css';

/**
 * Metric Card Component
 * Minimal borders, subtle depth, no thick shadows
 * Line-based icons only (removed emoji icons per design requirements)
 */
const MetricCard = ({ label, value, unit, color, trend, subtitle }) => {
  return (
    <div className={`metric-card ${color || ''}`}>
      <div className="metric-header">
        <div className="metric-label">{label}</div>
      </div>
      <div className="metric-content">
        <div className="metric-value">
          {value}
          {unit && <span className="metric-unit">{unit}</span>}
        </div>
        {subtitle && <div className="metric-subtitle">{subtitle}</div>}
        {trend && (
          <div className={`metric-trend ${trend > 0 ? 'positive' : 'negative'}`}>
            {trend > 0 ? '↑' : '↓'} {Math.abs(trend).toFixed(1)}%
          </div>
        )}
      </div>
    </div>
  );
};

export default MetricCard;
