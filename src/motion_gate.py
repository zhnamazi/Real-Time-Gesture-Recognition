"""
This module implements a motion detection system based on the velocity and acceleration of hand landmarks.
It uses an energy-based approach, calculating the sum of all landmark velocities to determine whether the hand is in motion or static.
The MotionGate class maintains a history of landmark positions and computes velocities and accelerations to classify the motion state.
"""

import numpy as np
from collections import deque

class MotionGate:
    """
    Motion detection based on velocity and acceleration of hand landmarks.
    Uses energy-based approach (sum of all landmark velocities).
    Attributes:
        buffer_size: Size of the history buffer
        velocity_threshold: Threshold for detecting motion based on velocity
        acceleration_threshold: Threshold for detecting motion based on acceleration
    """
    
    def __init__(self, buffer_size=15, velocity_threshold=0.02, acceleration_threshold=0.008):
        self.buffer_size = buffer_size
        self.velocity_threshold = velocity_threshold
        self.acceleration_threshold = acceleration_threshold
        
        # Store all 21 landmark positions
        self.landmarks_history = deque(maxlen=buffer_size)
        
        # Store computed velocities
        self.velocities = deque(maxlen=buffer_size)
    
    def update(self, landmarks):
        """
        Update the motion gate with new hand landmarks.
        Inputs:
            landmarks: List of hand landmarks (21 points)
        Returns:
            None
        """
        if landmarks is None or len(landmarks) < 21:
            return
        
        # Convert to numpy array for easier computation
        current_positions = np.array([(lm.x, lm.y, lm.z) for lm in landmarks])
        self.landmarks_history.append(current_positions)
        
        # Calculate velocity if we have previous frame
        if len(self.landmarks_history) >= 2:
            prev_positions = self.landmarks_history[-2]
            curr_positions = self.landmarks_history[-1]
            
            # Calculate displacement for each landmark
            displacements = curr_positions - prev_positions
            
            # Calculate velocity magnitude for each landmark (L2 norm per point)
            velocities_per_landmark = np.linalg.norm(displacements, axis=1)
            
            # Total motion energy: sum of all landmark velocities
            total_velocity = np.sum(velocities_per_landmark)
            
            self.velocities.append(total_velocity)
    
    def get_motion_type(self):
        """
        Determine if motion is static or dynamic
        If the average velocity is above the threshold, classify as dynamic; otherwise, static.
        Returns:
            'static' if motion is below thresholds, 'dynamic' if above thresholds, or 'unknown' if insufficient data
        """
        
        if len(self.landmarks_history) < 5:
            return 'unknown - low landmark history'
        
        if len(self.velocities) < 3:
            return 'unknown - low velocity history'
        
        # Calculate average velocity over buffer
        avg_velocity = np.mean(list(self.velocities))
        
        # Calculate acceleration (derivative of velocity)
        if len(self.velocities) >= 3:
            velocity_list = list(self.velocities)
            # Acceleration: change in velocity
            acceleration = np.std(velocity_list)  # Standard deviation of velocity
        else:
            acceleration = 0.0
        
        # Check for sustained motion (velocity + acceleration)
        # Both conditions must be met for "dynamic"
        has_velocity = avg_velocity > self.velocity_threshold
        has_acceleration = acceleration > self.acceleration_threshold
        
        # # Dynamic: sustained motion (both velocity and acceleration present)
        if has_velocity or has_acceleration:  # Either one is enough
            return 'dynamic'
        else:
            return 'static'

    def get_trajectory(self, method='wrist'):
        """
        Returns trajectory based on method
            method='wrist': wrist position (landmark 0)
            method='center': center of all landmarks (palm center)
            method='fingertips': average of fingertip positions (landmarks 4,8,12,16,20)
        Inputs:
            method: str, method to compute trajectory
        Returns:  
            trajectory: list of (x,y) tuples representing the trajectory
        """
        if len(self.landmarks_history) < 5:
            return None
        
        trajectory = []
        
        if method == 'wrist':
            # Wrist only (landmark 0)
            for frame_landmarks in self.landmarks_history:
                wrist_x = frame_landmarks[0][0]
                wrist_y = frame_landmarks[0][1]
                trajectory.append((wrist_x, wrist_y))
        
        elif method == 'center':
            # Center of all 21 landmarks (palm center + hand center)
            for frame_landmarks in self.landmarks_history:
                center_x = np.mean(frame_landmarks[:, 0])
                center_y = np.mean(frame_landmarks[:, 1])
                trajectory.append((center_x, center_y))
        
        elif method == 'fingertips':
            # Average of fingertip positions (landmarks 4,8,12,16,20)
            fingertip_indices = [4, 8, 12, 16, 20]
            for frame_landmarks in self.landmarks_history:
                fingertips = frame_landmarks[fingertip_indices]
                tips_x = np.mean(fingertips[:, 0])
                tips_y = np.mean(fingertips[:, 1])
                trajectory.append((tips_x, tips_y))
        
        return trajectory