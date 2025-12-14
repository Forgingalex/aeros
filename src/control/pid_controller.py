"""PID controller for heading correction."""

import time
from typing import Optional


class PIDController:
    """Proportional-Integral-Derivative controller for heading correction.
    
    Maintains desired heading by computing control output based on
    error between current and target heading.
    """
    
    def __init__(
        self,
        kp: float = 1.0,
        ki: float = 0.1,
        kd: float = 0.5,
        setpoint: float = 0.0,
        output_limits: Optional[tuple] = None,
    ):
        """Initialize PID controller.
        
        Args:
            kp: Proportional gain
            ki: Integral gain
            kd: Derivative gain
            setpoint: Target heading angle (radians)
            output_limits: Tuple of (min, max) output limits
        """
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.setpoint = setpoint
        self.output_limits = output_limits
        
        self._integral = 0.0
        self._last_error = 0.0
        self._last_time = None
    
    # TODO: Implement confidence-conditioned control: scale PID gains based on perception confidence.
    # When confidence < threshold (e.g., 0.7), reduce Kp/Kd to prevent overcorrection on ambiguous
    # visual features. Test hypothesis: confidence-based gain scheduling reduces oscillation in
    # low-texture environments (corridor walls) while maintaining responsiveness in high-texture
    # areas (doors, signs).
    
    def compute_control(self, current_heading: float) -> float:
        """Compute PID control output from heading error.
        
        Args:
            current_heading: Current heading angle (radians)
            
        Returns:
            Angular velocity command (rad/s)
        """
        error = self.setpoint - current_heading
        
        current_time = time.time()
        if self._last_time is None:
            self._last_time = current_time
            # Bug fix: initialize _last_error to prevent derivative term spike on first update
            # Without this, (error - undefined) / dt causes violent initial correction
            self._last_error = error
            dt = 0.01  # Default timestep
        else:
            dt = current_time - self._last_time
            if dt <= 0:
                dt = 0.01
        
        # Proportional term
        p_term = self.kp * error
        
        # Integral term
        self._integral += error * dt
        i_term = self.ki * self._integral
        
        # Derivative term
        if dt > 0:
            derivative = (error - self._last_error) / dt
        else:
            derivative = 0.0
        d_term = self.kd * derivative
        
        output = p_term + i_term + d_term
        
        # Apply output limits
        if self.output_limits is not None:
            output = max(self.output_limits[0], min(self.output_limits[1], output))
        
        # Update state
        self._last_error = error
        self._last_time = current_time
        
        return output
    
    def reset(self) -> None:
        """Reset controller state."""
        self._integral = 0.0
        self._last_error = 0.0
        self._last_time = None
    
    def set_setpoint(self, setpoint: float) -> None:
        """Update target setpoint.
        
        Args:
            setpoint: New target heading angle (radians)
        """
        self.setpoint = setpoint

