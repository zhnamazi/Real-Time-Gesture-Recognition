import sys
import cv2
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import mediapipe as mp
import pickle
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
from src.motion_gate import MotionGate

MODEL_PATH = "./src/hand_landmarker.task"
TEMPLATES_DIR = "./datasets/dynamic_templates"

# Create templates directory if it doesn't exist
os.makedirs(TEMPLATES_DIR, exist_ok=True)

GESTURE_NAMES = ["increase_volume", "decrease_volume", "bye"]
# Define trajectory method for each gesture
GESTURE_METHODS = {
    "increase_volume": "wrist",    # Wrist moves diagonally up
    "decrease_volume": "wrist",    # Wrist moves diagonally down
    "bye": "center"                # Palm center moves (hand shape changes, wrist stable)
}

def save_template(gesture_name, trajectories_dict, sample_num):
    """Save dict of 3 trajectories (wrist, center, fingertips)"""
    filename = f"{TEMPLATES_DIR}/{gesture_name}_{sample_num}.pkl"
    with open(filename, 'wb') as f:
        pickle.dump(trajectories_dict, f)
    num_frames = len(trajectories_dict['wrist'])
    print(f"Saved: {filename} (length: {num_frames} frames)")


def main():
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Cannot open camera")
        return
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    print("Loading Hand Landmarker...")
    try:
        base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=1,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5
        )
        detector = vision.HandLandmarker.create_from_options(options)
    except Exception as e:
        print(f"Error loading Hand Landmarker: {e}")
        return
    
    motion_gate = MotionGate(
        buffer_size=20,
        velocity_threshold=0.16,
        acceleration_threshold=0.6
    )
    
    # gesture_index = 0
    sample_count = {g: 0 for g in GESTURE_NAMES}
    
    print("\n" + "="*60)
    print("DYNAMIC GESTURE TEMPLATE CAPTURE")
    print("="*60)
    for i, gesture in enumerate(GESTURE_NAMES):
        method = GESTURE_METHODS[gesture]
        print(f"{i}: {gesture:20} (trajectory method: {method})")
    
    recording = False
    current_gesture = None
    current_trajectory = None
    
    print("\nKey commands:")
    print("  0-2: Select gesture (0=increase_volume, 1=decrease_volume, 2=bye)")
    print("  SPACE: Start/Stop recording")
    print("  s: Save the last recorded trajectory")
    print("  r: Reset (delete last trajectory without saving)")
    print("  ESC: Exit")
    print("="*60 + "\n")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        detection_result = detector.detect(mp_image)
        
        h, w, _ = frame.shape
        
        if len(detection_result.hand_landmarks) > 0:
            landmarks = detection_result.hand_landmarks[0]
            
            motion_gate.update(landmarks)
            # motion_type = motion_gate.get_motion_type()
            
            if recording and current_gesture is not None:
                # Collect trajectory while recording
                current_landmarks = np.array([(lm.x, lm.y, lm.z) for lm in landmarks])
                motion_gate.landmarks_history.append(current_landmarks)
            
            # Draw landmarks on frame
            for landmark in landmarks:
                x = int(landmark.x * w)
                y = int(landmark.y * h)
                cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)
            
            # # Draw wrist trajectory (last 20 frames)
            # method = GESTURE_METHODS[current_gesture]
            # trajectory = motion_gate.get_trajectory(method=method)
            # if trajectory and len(trajectory) > 1:
            #     for i in range(len(trajectory) - 1):
            #         x1 = int(trajectory[i][0] * w)
            #         y1 = int(trajectory[i][1] * h)
            #         x2 = int(trajectory[i+1][0] * w)
            #         y2 = int(trajectory[i+1][1] * h)
            #         cv2.line(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
        
        # Display status
        status_color = (0, 255, 0)
        if current_gesture is not None:
            gesture_text = f"Selected: {current_gesture}"
            cv2.putText(frame, gesture_text, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        if recording:
            status_text = "REC: Recording... (SPACE to stop, s to save, r to reset)"
            status_color = (0, 0, 255)
            if current_trajectory is not None:
                frame_text = f"Frames: {len(current_trajectory['wrist'])}"
                cv2.putText(frame, frame_text, (10, 90),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            status_text = "Press 0-2 to select gesture, SPACE to record"
        
        cv2.putText(frame, status_text, (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)
        
        # Display sample count
        counts_text = " | ".join([f"{g}: {sample_count[g]}" for g in GESTURE_NAMES])
        cv2.putText(frame, counts_text, (10, h - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        cv2.imshow('Dynamic Gesture Capture', frame)
        
        key = cv2.waitKey(1) & 0xFF
        
        if key == 27:  # ESC
            print("\nExiting...")
            break
        
        elif key in [ord('0'), ord('1'), ord('2')]:  # Select gesture
            gesture_idx = int(chr(key))
            if gesture_idx < len(GESTURE_NAMES):
                current_gesture = GESTURE_NAMES[gesture_idx]
                method = GESTURE_METHODS[current_gesture]
                print(f"\nSelected gesture: {current_gesture} (trajectory method: {method})")
                recording = False
                current_trajectory = None
        
        elif key == ord(' '):  # Start/Stop recording
            if current_gesture is None:
                print("Please select a gesture first (press 0-2)")
            else:
                if not recording:
                    # Start recording
                    recording = True
                    current_trajectory = None
                    motion_gate.landmarks_history.clear()
                    motion_gate.velocities.clear()
                    print(f"Started recording {current_gesture}...")
                else:
                    # Stop recording
                    recording = False
                    if len(motion_gate.landmarks_history) > 0:
                        current_trajectory = {
                            'wrist': motion_gate.get_trajectory(method='wrist'),
                            'center': motion_gate.get_trajectory(method='center'),
                            'fingertips': motion_gate.get_trajectory(method='fingertips')
                        }
                        num_frames = len(current_trajectory['wrist'])
                        print(f"Stopped recording. Trajectory has {num_frames} frames.")
        
        elif key == ord('s'):  # Save template
            if current_trajectory is not None and len(current_trajectory['wrist']) >= 10:
                sample_num = sample_count[current_gesture]
                save_template(current_gesture, current_trajectory, sample_num)
                sample_count[current_gesture] += 1
                current_trajectory = None
                print(f"Saved sample #{sample_num} for {current_gesture}")
            elif current_trajectory is None:
                print("No trajectory recorded yet. Record first (SPACE).")
            else:
                print(f"Trajectory too short ({len(current_trajectory['wrist'])} frames). Need at least 10.")        
        elif key == ord('r'):  # Reset last recording
            current_trajectory = None
            motion_gate.landmarks_history.clear()
            motion_gate.velocities.clear()
            print("Reset last recording.")
    
    cap.release()
    cv2.destroyAllWindows()
    
    print("\n" + "="*60)
    print("CAPTURE COMPLETE")
    print("="*60)
    for gesture in GESTURE_NAMES:
        method = GESTURE_METHODS[gesture]
        print(f"{gesture:20} (method: {method:10}): {sample_count[gesture]} samples")
    print("="*60)


if __name__ == "__main__":
    main()