"""
This module implements dynamic gesture detection using Dynamic Time Warping (DTW) to compare gesture trajectories against stored templates.
It supports multiple trajectory methods (wrist, center, fingertips) and provides visualization of both templates and current gestures.
"""

from matplotlib import pyplot as plt
import numpy as np
import pickle
import os

class DynamicGestureDetector:
    """
    Dynamic Gesture Detector using DTW
    Templates are expected as:
              - increase_volume_0.pkl, increase_volume_1.pkl, ...
              - decrease_volume_0.pkl, decrease_volume_1.pkl, ...
              - bye_0.pkl, bye_1.pkl, ...
    
    Attributes:
        templates_dir: directory containing gesture templates
    """
    
    def __init__(self, templates_dir="./src/artifacts/dynamic_templates"):
        self.templates_dir = templates_dir
        self.gesture_names = ["increase_volume", "decrease_volume", "bye"]
        self.templates = {name: [] for name in self.gesture_names}
        
        self.load_templates()
    
    def load_templates(self):
        """
        Load all template landmarks from directory
        
        Inputs:
            templates_dir: directory containing gesture templates

        Returns:
            None
        """
        if not os.path.exists(self.templates_dir):
            print(f"Warning: Templates directory not found: {self.templates_dir}")
            return
        
        for gesture_name in self.gesture_names:
            count = 0
            # Load all pkl files for this gesture
            for filename in sorted(os.listdir(self.templates_dir)):
                if filename.startswith(gesture_name) and filename.endswith('.pkl'):
                    filepath = os.path.join(self.templates_dir, filename)
                    try:
                        with open(filepath, 'rb') as f:
                            landmarks = pickle.load(f)
                            self.templates[gesture_name].append(landmarks)
                            count += 1
                    except Exception as e:
                        print(f"Error loading template {filepath}: {e}")
            
            if count > 0:
                print(f"Loaded {count} templates for {gesture_name}")
            else:
                print(f"No templates found for {gesture_name}")
    
    @staticmethod
    def normalize_trajectory(traj):
        """
        Translation normalization:
        Shift trajectory so that it starts at (0,0).
        
        Inputs:
            traj: numpy array of shape (N, 2) representing the trajectory points

        Returns:
            normalized_traj: numpy array of shape (N, 2) with the first point at (0,0)
        """
        traj = np.asarray(traj, dtype=np.float64)

        if len(traj) == 0:
            return traj

        return traj - traj[0]

    def visualize_templates(self, current_trajectory, normalize=False):
        """
        Visualize all stored templates grouped by gesture and trajectory method.
        
        Inputs:
            current_trajectory: dict with keys 'wrist', 'center', 'fingertips' containing the current gesture trajectory
            normalize: boolean indicating whether to normalize trajectories for visualization

        Returns:
            None
        """

        methods = ["wrist", "center", "fingertips"]

        fig, axes = plt.subplots(
            len(self.gesture_names) + 1,
            len(methods),
            figsize=(10, 8)
        )

        colors = ["r", "g", "b", "m", "c", "y", "k"]

        # Display trajectory of template gestures
        for row, gesture in enumerate(self.gesture_names):

            for col, method in enumerate(methods):

                ax = axes[row][col]

                templates = self.templates[gesture]

                for i, template_dict in enumerate(templates):

                    traj = template_dict[method]

                    if traj is None or len(traj) == 0:
                        continue

                    traj = np.array(traj, dtype=np.float64)
                    if normalize:
                        traj = self.normalize_trajectory(traj)

                    ax.plot(
                        traj[:,0],
                        traj[:,1],
                        color=colors[i % len(colors)],
                        linewidth=2,
                        label=f"T{i}"
                    )

                    # Draw start point
                    ax.scatter(
                        traj[0,0],
                        traj[0,1],
                        color=colors[i % len(colors)],
                        marker='o'
                    )

                    # Draw end point
                    ax.scatter(
                        traj[-1,0],
                        traj[-1,1],
                        color=colors[i % len(colors)],
                        marker='x'
                    )

                ax.set_title(f"{gesture}\n{method}")
                ax.set_aspect("equal")
                ax.invert_yaxis()
                ax.grid(True)

        # Display trajectory of real (current) gestures
        for col, method in enumerate(methods):
        
            ax = axes[-1][col]
            traj = current_trajectory[method]

            if traj is None or len(traj) == 0:
                continue

            traj = np.array(traj, dtype=np.float64)
            if normalize:
                traj = self.normalize_trajectory(traj)

            ax.plot(
                traj[:,0],
                traj[:,1],
                color=colors[(len(self.gesture_names) + 1) % len(colors)],
                linewidth=2,
                label=f"T{i}"
            )

            # Draw start point
            ax.scatter(
                traj[0,0],
                traj[0,1],
                color=colors[(len(self.gesture_names) + 1) % len(colors)],
                marker='o'
            )

            # Draw end point
            ax.scatter(
                traj[-1,0],
                traj[-1,1],
                color=colors[(len(self.gesture_names) + 1) % len(colors)],
                marker='x'
            )

            ax.set_title(f"current gesture\n{method}")
            ax.set_aspect("equal")
            ax.invert_yaxis()
            ax.grid(True)

            
        handles, labels = axes[0][0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper right")

        plt.tight_layout()
        plt.show()

    @staticmethod
    def trajectory_to_sequence(trajectory):
        """
        Convert trajectory (list of (x,y) tuples) to numpy array for DTW
        
        Inputs:
            trajectory: list of (x,y) tuples representing the gesture trajectory

        Returns:   
            numpy array of shape (N, 2) representing the trajectory points
        """

        if trajectory is None or len(trajectory) == 0:
            return None
        return np.array(trajectory, dtype=np.float32)
    
    @staticmethod
    def dtw_distance(traj1, traj2):
        """
        Calculate DTW distance between two trajectories
        Inputs:
            traj1: numpy array of shape (N, 2) representing the first trajectory
            traj2: numpy array of shape (M, 2) representing the second trajectory
        
        Returns:
            dtw_distance: float representing the DTW distance between the two trajectories (lower is more similar)
        """
        if traj1 is None or traj2 is None or len(traj1) < 3 or len(traj2) < 3:
            print("Warning: One or both trajectories are too short for DTW calculation.")
            return float('inf')
        
        traj1 = DynamicGestureDetector.normalize_trajectory(traj1)
        traj2 = DynamicGestureDetector.normalize_trajectory(traj2)

        n = len(traj1)
        m = len(traj2)

        dtw_matrix = np.full((n + 1, m + 1), np.inf)
        dtw_matrix[0, 0] = 0

        for i in range(1, n + 1):
            for j in range(1, m + 1):
                cost = np.linalg.norm(traj1[i-1] - traj2[j-1])

                dtw_matrix[i, j] = cost + min(
                    dtw_matrix[i-1, j],      # insertion
                    dtw_matrix[i, j-1],      # deletion
                    dtw_matrix[i-1, j-1]     # match
                )

        return dtw_matrix[n, m]
    
    @staticmethod
    def normalize_dtw_score(distance, traj_len):
        """
        Normalize DTW distance to get confidence [0, 1]
        Lower distance = higher confidence
        Uses path length as normalization factor
        
        Inputs:
            distance: float representing the DTW distance between two trajectories
            traj_len: int representing the length of the trajectory being evaluated
        
        Returns:
            confidence: float in [0, 1] representing the normalized confidence score
        """
        if distance == float('inf') or distance is None:
            return 0.0
        
        # Normalize by trajectory length (longer trajectories naturally have larger DTW distances)
        normalized = 1.0 / (1.0 + distance / (traj_len + 1e-6))
        return np.clip(normalized, 0.0, 1.0)
    
    def detect_gesture(self, trajectory_by_method, gesture_name):
        """
        Detect gesture using all 3 trajectory methods (ensemble)

        Inputs:
            trajectory_by_method: dict with keys 'wrist', 'center', 'fingertips' containing the current gesture trajectory
            gesture_name: string representing the name of the gesture to detect

        Returns:
            confidence: float in [0, 1] representing the normalized confidence score for the gesture
        """
        if trajectory_by_method is None:
            print(f"Trajectory too short for gesture {gesture_name}: length {len(trajectory)}")
            return False, 0.0
        
        if gesture_name not in self.templates or len(self.templates[gesture_name]) == 0:
            print(f"No templates available for gesture {gesture_name}")
            return False, 0.0

        dtw_scores = {}

        for method_name in ['wrist', 'center', 'fingertips']:
            trajectory = trajectory_by_method.get(method_name)
            if trajectory is None or len(trajectory) < 10:
                dtw_scores[method_name] = 0.0
                continue

            distances = []
            for template_dict in self.templates[gesture_name]:
                template_traj = template_dict[method_name]

                traj_seq = self.trajectory_to_sequence(trajectory)
                template_seq = self.trajectory_to_sequence(template_traj)

                if traj_seq is not None and template_seq is not None:
                    distance = self.dtw_distance(traj_seq, template_seq)
                    distances.append(distance)
        
            if distances and not all(d == float('inf') for d in distances):
                best_distance = min(distances)
                confidence = self.normalize_dtw_score(best_distance, len(trajectory))
                dtw_scores[method_name] = confidence
            else:
                dtw_scores[method_name] = 0.0
        
        # Combine scores from all methods (average)
        combined_confidence = np.mean(list(dtw_scores.values()))
        
        return combined_confidence
    
    def detect_best_match(self, trajectory_by_method):
        """
        Detect which gesture trajectory best matches (ensemble of all 3 methods)
        Inputs:
            trajectory_by_method: dict with keys 'wrist', 'center', 'fingertips' containing the current gesture trajectory
        
        Returns:
            best_gesture: string representing the name of the best matching gesture
            best_confidence: float representing the confidence score of the best match
        """
        best_gesture = "no_gesture"
        best_confidence = 0.0
        
        for gesture_name in self.gesture_names:
            confidence = self.detect_gesture(trajectory_by_method, gesture_name)
            if confidence > best_confidence:
                best_confidence = confidence
                best_gesture = gesture_name
        
        return best_gesture, best_confidence