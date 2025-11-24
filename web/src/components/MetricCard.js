import React from 'react';
import './MetricCard.css';

const MetricCard = ({ label, value, unit, icon, color, trend, subtitle }) => {
  return (
    <div className={`metric-card ${color || ''}`}>
      <div className="metric-header">
        {icon && <div className="metric-icon">{icon}</div>}
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

