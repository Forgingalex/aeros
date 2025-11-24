"""PyBullet simulation environment for drone navigation."""

import numpy as np
from typing import Tuple, Optional
import os
from pathlib import Path

# PyBullet is optional - only needed for simulation
try:
    import pybullet as p
    import pybullet_data
    PYBULLET_AVAILABLE = True
except ImportError:
    PYBULLET_AVAILABLE = False
    p = None
    pybullet_data = None


class DroneSimulation:
    """PyBullet simulation environment with drone and corridor.
    
    Creates a simulated drone with camera attached, flying through
    a corridor environment. Provides camera feed and control interface.
    """
    
    def __init__(
        self,
        gui: bool = True,
        corridor_length: float = 50.0,
        corridor_width: float = 4.0,
        corridor_height: float = 3.0,
    ):
        """Initialize simulation.
        
        Args:
            gui: Whether to show GUI
            corridor_length: Length of corridor (meters)
            corridor_width: Width of corridor (meters)
            corridor_height: Height of corridor (meters)
        
        Raises:
            ImportError: If PyBullet is not installed
        """
        if not PYBULLET_AVAILABLE:
            raise ImportError(
                "PyBullet is not installed. Install it with: pip install pybullet\n"
                "Note: PyBullet requires Visual C++ Build Tools on Windows."
            )
        
        self.gui = gui
        self.corridor_length = corridor_length
        self.corridor_width = corridor_width
        self.corridor_height = corridor_height
        
        # Initialize PyBullet
        if gui:
            self.client = p.connect(p.GUI)
        else:
            self.client = p.connect(p.DIRECT)
        
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        
        # Create environment
        self._create_corridor()
        self._load_drone()
        
        # Camera parameters
        self.camera_width = 224
        self.camera_height = 224
        self.camera_fov = 60
        self.camera_near = 0.1
        self.camera_far = 100.0
        
        # Drone state
        self.drone_position = np.array([0.0, 0.0, 1.5])
        self.drone_orientation = np.array([0.0, 0.0, 0.0, 1.0])  # quaternion
        
    def _create_corridor(self) -> None:
        """Create corridor environment."""
        # Floor
        floor_shape = p.createCollisionShape(
            p.GEOM_BOX,
            halfExtents=[self.corridor_length / 2, self.corridor_width / 2, 0.1],
        )
        floor_visual = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=[self.corridor_length / 2, self.corridor_width / 2, 0.1],
            rgbaColor=[0.5, 0.5, 0.5, 1.0],
        )
        p.createMultiBody(
            0, floor_shape, floor_visual, [0, 0, 0]
        )
        
        # Left wall
        wall_shape = p.createCollisionShape(
            p.GEOM_BOX,
            halfExtents=[self.corridor_length / 2, 0.1, self.corridor_height / 2],
        )
        wall_visual = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=[self.corridor_length / 2, 0.1, self.corridor_height / 2],
            rgbaColor=[0.7, 0.7, 0.7, 1.0],
        )
        p.createMultiBody(
            0, wall_shape, wall_visual,
            [0, -self.corridor_width / 2, self.corridor_height / 2]
        )
        
        # Right wall
        p.createMultiBody(
            0, wall_shape, wall_visual,
            [0, self.corridor_width / 2, self.corridor_height / 2]
        )
        
        # Ceiling
        ceiling_shape = p.createCollisionShape(
            p.GEOM_BOX,
            halfExtents=[self.corridor_length / 2, self.corridor_width / 2, 0.1],
        )
        ceiling_visual = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=[self.corridor_length / 2, self.corridor_width / 2, 0.1],
            rgbaColor=[0.5, 0.5, 0.5, 1.0],
        )
        p.createMultiBody(
            0, ceiling_shape, ceiling_visual,
            [0, 0, self.corridor_height]
        )
    
    def _load_drone(self) -> None:
        """Load drone model."""
        # Create simple box drone
        drone_shape = p.createCollisionShape(
            p.GEOM_BOX, halfExtents=[0.2, 0.2, 0.1]
        )
        drone_visual = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=[0.2, 0.2, 0.1],
            rgbaColor=[0.2, 0.8, 0.2, 1.0],
        )
        
        self.drone_id = p.createMultiBody(
            0.5,  # mass
            drone_shape,
            drone_visual,
            basePosition=self.drone_position.tolist(),
            baseOrientation=self.drone_orientation.tolist(),
        )
        
        # Set initial velocity
        p.resetBaseVelocity(self.drone_id, [1.0, 0.0, 0.0], [0.0, 0.0, 0.0])
    
    def get_camera_image(self) -> np.ndarray:
        """Get camera image from drone's perspective.
        
        Returns:
            RGB image array (H, W, 3)
        """
        # Get drone position and orientation
        pos, orn = p.getBasePositionAndOrientation(self.drone_id)
        
        # Convert quaternion to rotation matrix
        rot_matrix = p.getMatrixFromQuaternion(orn)
        forward = np.array([rot_matrix[0], rot_matrix[3], rot_matrix[6]])
        up = np.array([rot_matrix[2], rot_matrix[5], rot_matrix[8]])
        
        # Camera position (slightly forward from drone center)
        camera_pos = np.array(pos) + forward * 0.1
        
        # Camera target (looking forward)
        target_pos = camera_pos + forward * 5.0
        
        # Get camera image
        view_matrix = p.computeViewMatrix(
            cameraEyePosition=camera_pos,
            cameraTargetPosition=target_pos,
            cameraUpVector=up,
        )
        
        projection_matrix = p.computeProjectionMatrixFOV(
            fov=self.camera_fov,
            aspect=float(self.camera_width) / self.camera_height,
            nearVal=self.camera_near,
            farVal=self.camera_far,
        )
        
        _, _, rgba, depth, _ = p.getCameraImage(
            width=self.camera_width,
            height=self.camera_height,
            viewMatrix=view_matrix,
            projectionMatrix=projection_matrix,
        )
        
        # Convert RGBA to RGB
        rgb = rgba[:, :, :3]
        return rgb
    
    def apply_control(self, angular_velocity: float) -> None:
        """Apply control command to drone.
        
        Args:
            angular_velocity: Angular velocity command (rad/s)
        """
        # Get current state
        pos, orn = p.getBasePositionAndOrientation(self.drone_id)
        vel, ang_vel = p.getBaseVelocity(self.drone_id)
        
        # Apply angular velocity (yaw control)
        new_ang_vel = list(ang_vel)
        new_ang_vel[2] = angular_velocity  # Z-axis rotation (yaw)
        
        # Maintain forward velocity
        forward_vel = np.array([1.0, 0.0, 0.0])
        rot_matrix = p.getMatrixFromQuaternion(orn)
        forward = np.array([rot_matrix[0], rot_matrix[3], rot_matrix[6]])
        right = np.array([rot_matrix[1], rot_matrix[4], rot_matrix[7]])
        
        # Rotate forward vector by yaw
        yaw_angle = np.arctan2(forward[1], forward[0])
        new_yaw = yaw_angle + angular_velocity * 0.01
        
        new_forward = np.array([np.cos(new_yaw), np.sin(new_yaw), 0.0])
        new_vel = new_forward * np.linalg.norm(forward_vel)
        
        p.resetBaseVelocity(self.drone_id, new_vel.tolist(), new_ang_vel)
    
    def step(self, dt: float = 1.0 / 60.0) -> None:
        """Step simulation forward.
        
        Args:
            dt: Time step (seconds)
        """
        p.stepSimulation()
        if self.gui:
            import time
            time.sleep(dt)
    
    def get_drone_state(self) -> dict:
        """Get current drone state.
        
        Returns:
            Dictionary with position, orientation, velocity, etc.
        """
        pos, orn = p.getBasePositionAndOrientation(self.drone_id)
        vel, ang_vel = p.getBaseVelocity(self.drone_id)
        
        # Compute heading angle
        rot_matrix = p.getMatrixFromQuaternion(orn)
        forward = np.array([rot_matrix[0], rot_matrix[3], rot_matrix[6]])
        heading = np.arctan2(forward[1], forward[0])
        
        return {
            "position": pos,
            "orientation": orn,
            "velocity": vel,
            "angular_velocity": ang_vel,
            "heading": heading,
        }
    
    def reset(self) -> None:
        """Reset simulation to initial state."""
        p.resetBasePositionAndOrientation(
            self.drone_id,
            self.drone_position.tolist(),
            self.drone_orientation.tolist(),
        )
        p.resetBaseVelocity(self.drone_id, [1.0, 0.0, 0.0], [0.0, 0.0, 0.0])
    
    def close(self) -> None:
        """Close simulation."""
        p.disconnect(self.client)

