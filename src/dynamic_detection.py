import numpy as np
import pickle
import os
from dtaidistance import dtw

class DynamicGestureDetector:
    """Detect dynamic gestures using DTW template matching with gesture-specific trajectory methods"""
    
    def __init__(self, templates_dir="./datasets/dynamic_templates"):
        """
        Load gesture templates from directory
        Templates are expected as:
          - increase_volume_0.pkl, increase_volume_1.pkl, ...
          - decrease_volume_0.pkl, decrease_volume_1.pkl, ...
          - bye_0.pkl, bye_1.pkl, ...
        
        Trajectory methods per gesture:
          - increase_volume: 'wrist' (wrist moves diagonally up)
          - decrease_volume: 'wrist' (wrist moves diagonally down)
          - bye: 'center' (palm center trajectory; hand shape changes, wrist stable)
        
        To test different methods, simply change the value in self.trajectory_methods dict.
        """
        self.templates_dir = templates_dir
        self.gesture_names = ["increase_volume", "decrease_volume", "bye"]
        self.templates = {name: [] for name in self.gesture_names}
        
        # Define which trajectory method to use for each gesture
        # EDIT THESE TO TEST DIFFERENT METHODS:
        self.trajectory_methods = {
            "increase_volume": "wrist",      # Motion-based gestures use wrist
            "decrease_volume": "wrist",      # Motion-based gestures use wrist
            "bye": "center"                  # Shape-based gesture uses palm center
            # To test fingertips for bye, change to: "bye": "fingertips"
        }
        
        self.dtw_threshold = 0.5  # Threshold for confidence (normalized DTW distance)
        
        self.load_templates()
    
    def load_templates(self):
        """Load all template trajectories from directory"""
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
                            trajectory = pickle.load(f)
                            self.templates[gesture_name].append(trajectory)
                            count += 1
                    except Exception as e:
                        print(f"Error loading template {filepath}: {e}")
            
            method = self.trajectory_methods.get(gesture_name, "wrist")
            if count > 0:
                print(f"Loaded {count} templates for {gesture_name} (method: {method})")
            else:
                print(f"No templates found for {gesture_name}")
    
    @staticmethod
    def trajectory_to_sequence(trajectory):
        """Convert trajectory (list of (x,y) tuples) to numpy array for DTW"""
        if trajectory is None or len(trajectory) == 0:
            return None
        return np.array(trajectory, dtype=np.float32)
    
    @staticmethod
    def dtw_distance(traj1, traj2):
        """
        Calculate DTW distance between two trajectories
        Returns: distance (lower is better)
        """
        if traj1 is None or traj2 is None or len(traj1) < 3 or len(traj2) < 3:
            print("Warning: One or both trajectories are too short for DTW calculation.")
            return float('inf')
        
        try:
            # Ensure C-contiguous float64 arrays for dtaidistance
            # Extract x and y coordinates
            x1 = np.array([p[0] for p in traj1], dtype=np.float64)
            y1 = np.array([p[1] for p in traj1], dtype=np.float64)
            
            x2 = np.array([p[0] for p in traj2], dtype=np.float64)
            y2 = np.array([p[1] for p in traj2], dtype=np.float64)
            
            # Calculate DTW for each dimension
            distance_x = dtw.distance(x1, x2)
            distance_y = dtw.distance(y1, y2)
            
            # Average or weighted combination
            distance = (distance_x + distance_y) / 2.0
            
            return distance
        except Exception as e:
            print(f"DTW error: {type(e).__name__}: {e}")
            print(f"  traj1 shape: {traj1.shape}, dtype: {traj1.dtype}")
            print(f"  traj2 shape: {traj2.shape}, dtype: {traj2.dtype}")
            return float('inf')
    
    @staticmethod
    def normalize_dtw_score(distance, traj_len):
        """
        Normalize DTW distance to get confidence [0, 1]
        Lower distance = higher confidence
        Uses path length as normalization factor
        """
        if distance == float('inf') or distance is None:
            return 0.0
        
        # Normalize by trajectory length (longer trajectories naturally have larger DTW distances)
        normalized = 1.0 / (1.0 + distance / (traj_len + 1e-6))
        return np.clip(normalized, 0.0, 1.0)
    
    def detect_gesture(self, trajectory_by_method, gesture_name):
        """
        Detect gesture using all 3 trajectory methods (ensemble)
        trajectory_by_method: dict with 'wrist', 'center', 'fingertips'
        Returns: (is_detected, confidence)
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
        is_detected = combined_confidence > self.dtw_threshold
        
        return is_detected, combined_confidence
    
    def detect_increase_volume(self, trajectory):
        """Detect increase_volume gesture using DTW (wrist trajectory)"""
        return self.detect_gesture(trajectory, "increase_volume")
    
    def detect_decrease_volume(self, trajectory):
        """Detect decrease_volume gesture using DTW (wrist trajectory)"""
        return self.detect_gesture(trajectory, "decrease_volume")
    
    def detect_bye(self, trajectory):
        """Detect bye gesture using DTW (palm center trajectory)"""
        return self.detect_gesture(trajectory, "bye")
    
    def detect_best_match(self, trajectory_by_method):
        """
        Detect which gesture trajectory best matches (ensemble of all 3 methods)
        trajectory_by_method: dict with keys 'wrist', 'center', 'fingertips'
        Returns: (gesture_name, confidence)
        """
        best_gesture = "no_gesture"
        best_confidence = 0.0
        
        for gesture_name in self.gesture_names:
            is_detected, confidence = self.detect_gesture(trajectory_by_method, gesture_name)
            # print(f"Gesture: {gesture_name}, Method: {method}, Detected: {is_detected}, Confidence: {confidence:.3f}")
            if confidence > best_confidence:
                best_confidence = confidence
                best_gesture = gesture_name
        
        return best_gesture, best_confidence