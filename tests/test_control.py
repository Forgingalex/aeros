"""Tests for PID controller."""

import pytest
from src.control.pid_controller import PIDController


def test_pid_initialization():
    """Test PID controller initialization."""
    pid = PIDController(kp=1.0, ki=0.1, kd=0.5)
    assert pid.kp == 1.0
    assert pid.ki == 0.1
    assert pid.kd == 0.5
    assert pid.setpoint == 0.0


def test_pid_compute_control():
    """Test PID controller compute_control."""
    pid = PIDController(kp=1.0, ki=0.1, kd=0.5, setpoint=0.0)
    
    # Test with error
    output = pid.compute_control(0.5)  # Current heading is 0.5, setpoint is 0.0
    
    # Output should be negative (to reduce error)
    assert output < 0


def test_pid_reset():
    """Test PID controller reset."""
    pid = PIDController()
    
    # Update a few times
    pid.compute_control(0.5)
    pid.compute_control(0.3)
    
    # Reset
    pid.reset()
    
    # Check state is reset
    assert pid._integral == 0.0
    assert pid._last_error == 0.0


def test_pid_output_limits():
    """Test PID controller output limits."""
    pid = PIDController(
        kp=10.0,
        output_limits=(-1.0, 1.0)
    )
    
    # Large error should be clamped
    output = pid.compute_control(10.0)
    
    assert -1.0 <= output <= 1.0

