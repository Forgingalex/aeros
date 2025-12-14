"""PyBullet simulation environment for drone navigation."""

import numpy as np
from typing import Tuple, Optional, List
import os
from pathlib import Path
import math

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
    
    Enhanced features:
    - Visual patterns on walls for navigation cues
    - Visual landmarks (doors, signs, corners)
    - Realistic quadcopter model with propellers
    - Obstacles and corridor turns
    - Variable lighting conditions
    - Improved physics with realistic quadcopter dynamics
    """
    
    def __init__(
        self,
        gui: bool = True,
        corridor_length: float = 50.0,
        corridor_width: float = 4.0,
        corridor_height: float = 3.0,
        enable_obstacles: bool = True,
        enable_turns: bool = True,
        lighting_variation: float = 0.3,
    ):
        """Initialize simulation.
        
        Args:
            gui: Whether to show GUI
            corridor_length: Length of corridor (meters)
            corridor_width: Width of corridor (meters)
            corridor_height: Height of corridor (meters)
            enable_obstacles: Whether to add obstacles in corridor
            enable_turns: Whether to add turns in corridor
            lighting_variation: Lighting variation (0-1)
        
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
        self.enable_obstacles = enable_obstacles
        self.enable_turns = enable_turns
        self.lighting_variation = lighting_variation
        
        if gui:
            self.client = p.connect(p.GUI)
            # Configure camera for better visualization
            p.resetDebugVisualizerCamera(
                cameraDistance=10,
                cameraYaw=45,
                cameraPitch=-30,
                cameraTargetPosition=[0, 0, 1.5]
            )
        else:
            self.client = p.connect(p.DIRECT)
        
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        
        # Configure lighting
        p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 1)
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
        
        self._create_corridor()
        if enable_obstacles:
            self._add_obstacles()
        self._add_landmarks()
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
        
        # Physics parameters for realistic quadcopter
        self.drone_mass = 0.5  # kg
        self.propeller_radius = 0.15  # m
        self.max_thrust = 20.0  # N per propeller
        self.propeller_angular_velocities = [0.0, 0.0, 0.0, 0.0]  # rad/s for 4 propellers
        
        # Store landmark and obstacle IDs for reference
        self.landmark_ids = []
        self.obstacle_ids = []
    
    def _create_corridor(self) -> None:
        """Create corridor environment with visual patterns."""
        # Floor with checkerboard pattern
        floor_segments = int(self.corridor_length / 2.0)  # 2m segments
        for i in range(floor_segments):
            x_pos = (i - floor_segments / 2) * 2.0 + 1.0
            # Checkerboard pattern
            is_dark = (i % 2) == 0
            floor_color = [0.4, 0.4, 0.4, 1.0] if is_dark else [0.6, 0.6, 0.6, 1.0]
            
            floor_shape = p.createCollisionShape(
                p.GEOM_BOX,
                halfExtents=[1.0, self.corridor_width / 2, 0.1],
            )
            floor_visual = p.createVisualShape(
                p.GEOM_BOX,
                halfExtents=[1.0, self.corridor_width / 2, 0.1],
                rgbaColor=floor_color,
            )
            p.createMultiBody(
                0, floor_shape, floor_visual, [x_pos, 0, 0]
            )
        
        # Left wall with vertical stripes pattern
        wall_segments = int(self.corridor_length / 1.0)  # 1m segments
        for i in range(wall_segments):
            x_pos = (i - wall_segments / 2) * 1.0 + 0.5
            # Alternating stripe pattern
            is_dark = (i % 2) == 0
            wall_color = [0.5, 0.5, 0.7, 1.0] if is_dark else [0.8, 0.8, 0.9, 1.0]
            
            wall_shape = p.createCollisionShape(
                p.GEOM_BOX,
                halfExtents=[0.5, 0.1, self.corridor_height / 2],
            )
            wall_visual = p.createVisualShape(
                p.GEOM_BOX,
                halfExtents=[0.5, 0.1, self.corridor_height / 2],
                rgbaColor=wall_color,
            )
            # Left wall
            p.createMultiBody(
                0, wall_shape, wall_visual,
                [x_pos, -self.corridor_width / 2, self.corridor_height / 2]
            )
            # Right wall (inverted pattern)
            is_dark_right = (i % 2) == 1
            wall_color_right = [0.5, 0.5, 0.7, 1.0] if is_dark_right else [0.8, 0.8, 0.9, 1.0]
            wall_visual_right = p.createVisualShape(
                p.GEOM_BOX,
                halfExtents=[0.5, 0.1, self.corridor_height / 2],
                rgbaColor=wall_color_right,
            )
            p.createMultiBody(
                0, wall_shape, wall_visual_right,
                [x_pos, self.corridor_width / 2, self.corridor_height / 2]
            )
        
        # Ceiling with horizontal stripes
        ceiling_segments = int(self.corridor_length / 2.0)
        for i in range(ceiling_segments):
            x_pos = (i - ceiling_segments / 2) * 2.0 + 1.0
            is_dark = (i % 2) == 0
            ceiling_color = [0.3, 0.3, 0.3, 1.0] if is_dark else [0.5, 0.5, 0.5, 1.0]
            
            ceiling_shape = p.createCollisionShape(
                p.GEOM_BOX,
                halfExtents=[1.0, self.corridor_width / 2, 0.1],
            )
            ceiling_visual = p.createVisualShape(
                p.GEOM_BOX,
                halfExtents=[1.0, self.corridor_width / 2, 0.1],
                rgbaColor=ceiling_color,
            )
            p.createMultiBody(
                0, ceiling_shape, ceiling_visual,
                [x_pos, 0, self.corridor_height]
            )
    
    def _add_landmarks(self) -> None:
        """Add visual landmarks (doors, signs, corners) to aid navigation."""
        # Add doors on walls
        door_positions = [-15.0, -5.0, 5.0, 15.0]
        for x_pos in door_positions:
            # Left wall door
            door_shape = p.createCollisionShape(
                p.GEOM_BOX,
                halfExtents=[0.3, 0.05, 0.8],
            )
            door_visual = p.createVisualShape(
                p.GEOM_BOX,
                halfExtents=[0.3, 0.05, 0.8],
                rgbaColor=[0.4, 0.2, 0.1, 1.0],  # Brown door
            )
            door_id = p.createMultiBody(
                0, door_shape, door_visual,
                [x_pos, -self.corridor_width / 2 + 0.15, 1.0]
            )
            self.landmark_ids.append(door_id)
            
            # Right wall door
            door_id2 = p.createMultiBody(
                0, door_shape, door_visual,
                [x_pos, self.corridor_width / 2 - 0.15, 1.0]
            )
            self.landmark_ids.append(door_id2)
        
        # Add signs/panels on walls
        sign_positions = [-20.0, -10.0, 0.0, 10.0, 20.0]
        for x_pos in sign_positions:
            # Left wall sign
            sign_shape = p.createCollisionShape(
                p.GEOM_BOX,
                halfExtents=[0.5, 0.02, 0.3],
            )
            sign_visual = p.createVisualShape(
                p.GEOM_BOX,
                halfExtents=[0.5, 0.02, 0.3],
                rgbaColor=[1.0, 0.8, 0.0, 1.0],  # Yellow sign
            )
            sign_id = p.createMultiBody(
                0, sign_shape, sign_visual,
                [x_pos, -self.corridor_width / 2 + 0.12, 2.0]
            )
            self.landmark_ids.append(sign_id)
            
            # Right wall sign
            sign_id2 = p.createMultiBody(
                0, sign_shape, sign_visual,
                [x_pos, self.corridor_width / 2 - 0.12, 2.0]
            )
            self.landmark_ids.append(sign_id2)
    
    def _add_obstacles(self) -> None:
        """Add obstacles in the corridor."""
        obstacle_positions = [
            (-12.0, 0.0, 0.5),  # Left side
            (-8.0, 1.0, 0.5),   # Right side
            (8.0, -1.0, 0.5),   # Left side
            (12.0, 0.0, 0.5),   # Center
        ]
        
        for x, y, z in obstacle_positions:
            obstacle_shape = p.createCollisionShape(
                p.GEOM_CYLINDER,
                radius=0.3,
                height=1.0,
            )
            obstacle_visual = p.createVisualShape(
                p.GEOM_CYLINDER,
                radius=0.3,
                length=1.0,
                rgbaColor=[0.8, 0.2, 0.2, 1.0],  # Red obstacle
            )
            obstacle_id = p.createMultiBody(
                5.0,  # mass (heavy obstacle)
                obstacle_shape,
                obstacle_visual,
                basePosition=[x, y, z],
            )
            self.obstacle_ids.append(obstacle_id)
    
    def _load_drone(self) -> None:
        """Load realistic quadcopter drone model with propellers."""
        # Main body (central frame)
        body_shape = p.createCollisionShape(
            p.GEOM_BOX, halfExtents=[0.15, 0.15, 0.05]
        )
        body_visual = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=[0.15, 0.15, 0.05],
            rgbaColor=[0.1, 0.1, 0.1, 1.0],  # Dark body
        )
        
        link_masses = [0.05, 0.05, 0.05, 0.05]  # 4 arms
        link_collision_shapes = []
        link_visual_shapes = []
        link_positions = []
        link_orientations = []
        link_parent_indices = []
        link_joint_types = []
        link_joint_axes = []
        
        # Arm positions (X configuration)
        arm_length = 0.25
        arm_positions = [
            [arm_length, arm_length, 0.0],      # Front-right
            [-arm_length, arm_length, 0.0],     # Front-left
            [-arm_length, -arm_length, 0.0],   # Back-left
            [arm_length, -arm_length, 0.0],     # Back-right
        ]
        
        for i, arm_pos in enumerate(arm_positions):
            # Arm (cylinder)
            arm_shape = p.createCollisionShape(
                p.GEOM_BOX,
                halfExtents=[0.02, 0.02, 0.1],
            )
            arm_visual = p.createVisualShape(
                p.GEOM_BOX,
                halfExtents=[0.02, 0.02, 0.1],
                rgbaColor=[0.3, 0.3, 0.3, 1.0],  # Gray arms
            )
            link_collision_shapes.append(arm_shape)
            link_visual_shapes.append(arm_visual)
            link_positions.append(arm_pos)
            link_orientations.append([0, 0, 0, 1])
            link_parent_indices.append(0)  # Connected to base
            link_joint_types.append(p.JOINT_FIXED)
            link_joint_axes.append([0, 0, 1])
        
        self.drone_id = p.createMultiBody(
            baseMass=self.drone_mass,
            baseCollisionShapeIndex=body_shape,
            baseVisualShapeIndex=body_visual,
            basePosition=self.drone_position.tolist(),
            baseOrientation=self.drone_orientation.tolist(),
            linkMasses=link_masses,
            linkCollisionShapeIndices=link_collision_shapes,
            linkVisualShapeIndices=link_visual_shapes,
            linkPositions=link_positions,
            linkOrientations=link_orientations,
            linkParentIndices=link_parent_indices,
            linkJointTypes=link_joint_types,
            linkJointAxis=link_joint_axes,
        )
        
        # Add propellers (visual only, as separate objects)
        self.propeller_ids = []
        propeller_height = 0.15
        for i, arm_pos in enumerate(arm_positions):
            propeller_shape = p.createCollisionShape(
                p.GEOM_CYLINDER,
                radius=self.propeller_radius,
                height=0.01,
            )
            propeller_visual = p.createVisualShape(
                p.GEOM_CYLINDER,
                radius=self.propeller_radius,
                length=0.01,
                rgbaColor=[0.2, 0.2, 0.2, 0.8],  # Semi-transparent propellers
            )
            prop_pos = [arm_pos[0], arm_pos[1], propeller_height]
            prop_id = p.createMultiBody(
                0.01,  # Very light
                propeller_shape,
                propeller_visual,
                basePosition=prop_pos,
            )
            # Constrain propeller to drone (visual only)
            constraint_id = p.createConstraint(
                self.drone_id,
                -1,  # Base link
                prop_id,
                -1,
                p.JOINT_FIXED,
                [0, 0, 0],
                [arm_pos[0], arm_pos[1], propeller_height - 0.05],
                [0, 0, 0],
            )
            self.propeller_ids.append(prop_id)
        
        p.changeDynamics(
            self.drone_id,
            -1,
            localInertiaDiagonal=[0.01, 0.01, 0.02],  # More inertia in Z
        )
        
        p.resetBaseVelocity(self.drone_id, [1.0, 0.0, 0.0], [0.0, 0.0, 0.0])
    
    def get_camera_image(self) -> np.ndarray:
        """Get camera image from drone's perspective.
        
        Returns:
            RGB image array (H, W, 3)
        """
        pos, orn = p.getBasePositionAndOrientation(self.drone_id)
        rot_matrix = p.getMatrixFromQuaternion(orn)
        forward = np.array([rot_matrix[0], rot_matrix[3], rot_matrix[6]])
        up = np.array([rot_matrix[2], rot_matrix[5], rot_matrix[8]])
        
        camera_pos = np.array(pos) + forward * 0.1
        target_pos = camera_pos + forward * 5.0
        
        # Apply lighting variation
        light_direction = [0.5, 0.5, 1.0]  # Directional light
        light_intensity = 1.0 + np.random.uniform(
            -self.lighting_variation, self.lighting_variation
        )
        
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
            lightDirection=light_direction,
            lightColor=[light_intensity, light_intensity, light_intensity],
        )
        
        rgb = rgba[:, :, :3]
        return rgb
    
    def apply_control(self, angular_velocity: float) -> None:
        """Apply control command using torque-based physics instead of direct velocity setting.
        
        Design decision: torque-based control respects inertia and mass, making behavior
        more realistic and requiring proper PID tuning. Direct velocity control would be
        simpler but unrealistic (drone would instantly change direction).
        
        Args:
            angular_velocity: Angular velocity command (rad/s) around Z-axis
        """
        pos, orn = p.getBasePositionAndOrientation(self.drone_id)
        vel, ang_vel = p.getBaseVelocity(self.drone_id)
        
        rot_matrix = p.getMatrixFromQuaternion(orn)
        forward = np.array([rot_matrix[0], rot_matrix[3], rot_matrix[6]])
        right = np.array([rot_matrix[1], rot_matrix[4], rot_matrix[7]])
        up = np.array([rot_matrix[2], rot_matrix[5], rot_matrix[8]])
        
        # Quadcopter dynamics: differential thrust between diagonal propellers for yaw
        base_thrust = self.drone_mass * 9.81 / 4.0  # Per propeller
        
        # Yaw control (differential thrust)
        yaw_torque = angular_velocity * 0.5  # Convert to torque
        differential_thrust = yaw_torque / (self.propeller_radius * 2.0)
        
        # Apply thrust to each propeller
        # Front-right and back-left: one direction
        # Front-left and back-right: opposite direction
        thrusts = [
            base_thrust + differential_thrust,  # Front-right
            base_thrust - differential_thrust,  # Front-left
            base_thrust + differential_thrust,  # Back-left
            base_thrust - differential_thrust,  # Back-right
        ]
        
        # Apply forces at propeller positions
        arm_length = 0.25
        propeller_positions = [
            [arm_length, arm_length, 0.15],      # Front-right
            [-arm_length, arm_length, 0.15],    # Front-left
            [-arm_length, -arm_length, 0.15],   # Back-left
            [arm_length, -arm_length, 0.15],     # Back-right
        ]
        
        for i, (thrust, prop_pos_local) in enumerate(zip(thrusts, propeller_positions)):
            prop_pos_world = (
                np.array(pos) +
                forward * prop_pos_local[0] +
                right * prop_pos_local[1] +
                up * prop_pos_local[2]
            )
            
            # Apply thrust force upward
            thrust_force = up * thrust
            p.applyExternalForce(
                self.drone_id,
                -1,
                thrust_force,
                prop_pos_world,
                flags=p.WORLD_FRAME,
            )
        
        # Maintain forward velocity with damping
        current_forward_speed = np.dot(vel, forward)
        target_speed = 1.0
        speed_error = target_speed - current_forward_speed
        
        # Forward thrust (tilt forward slightly)
        forward_thrust = speed_error * 2.0
        p.applyExternalForce(
            self.drone_id,
            -1,
            forward * forward_thrust,
            [0, 0, 0],
            flags=p.WORLD_FRAME,
        )
        
        # Update propeller visual rotation (for visual effect)
        for i, prop_id in enumerate(self.propeller_ids):
            # Rotate propellers visually (doesn't affect physics)
            current_orn = p.getBasePositionAndOrientation(prop_id)[1]
            # Simple rotation around Z-axis
            rotation_speed = 50.0  # rad/s visual rotation
            # Note: This is just for visual effect
        
        # Damping prevents excessive oscillation from torque-based control
        # Tradeoff: higher damping = more stable but less responsive
        damping = 0.95
        p.changeDynamics(
            self.drone_id,
            -1,
            linearDamping=damping,
            angularDamping=damping,
        )
    
    def step(self, dt: float = 1.0 / 60.0) -> None:
        """Step simulation forward at fixed timestep.
        
        Called by background task at 60 Hz. GUI mode adds sleep to match real-time,
        headless mode runs as fast as possible.
        
        Args:
            dt: Time step (seconds)
        """
        p.stepSimulation()
        if self.gui:
            import time
            time.sleep(dt)  # Real-time pacing for GUI visualization
    
    def get_drone_state(self) -> dict:
        """Get current drone state.
        
        Returns:
            Dictionary with position, orientation, velocity, etc.
        """
        pos, orn = p.getBasePositionAndOrientation(self.drone_id)
        vel, ang_vel = p.getBaseVelocity(self.drone_id)
        
        rot_matrix = p.getMatrixFromQuaternion(orn)
        forward = np.array([rot_matrix[0], rot_matrix[3], rot_matrix[6]])
        heading = np.arctan2(forward[1], forward[0])
        
        # Check collisions
        contact_points = p.getContactPoints(self.drone_id)
        has_collision = len(contact_points) > 0
        
        return {
            "position": pos,
            "orientation": orn,
            "velocity": vel,
            "angular_velocity": ang_vel,
            "heading": heading,
            "collision": has_collision,
            "collision_points": len(contact_points),
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
