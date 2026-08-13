"""
Main script for real-time hand gesture recognition using MediaPipe Hand Landmarker and a trained classifier.
This script captures hand gestures from a webcam, classifies them as static or dynamic, and displays the results in real-time.
"""

import math
import cv2
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import mediapipe as mp
import pickle
import time
from collections import deque
from motion_gate import MotionGate
from dynamic_detection import DynamicGestureDetector

MODEL_PATH = "./src/artifacts/hand_landmarker.task"
CLASSIFIER_PATH = "./src/artifacts/gesture_classifier_mlp.pkl"
SCALER_PATH = "./src/artifacts/gesture_scaler.pkl"

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
    (5, 9), (9, 13), (13, 17)
]


def draw_landmarks_on_image(rgb_image, detection_result):
    """
    Draw hand landmarks and connections on the image.
    Inputs:
        rgb_image: numpy array representing the RGB image
        detection_result: MediaPipe HandLandmarker result containing hand landmarks and handedness
    Returns:
        annotated_image: numpy array representing the image with drawn landmarks and connections
    """
    hand_landmarks_list = detection_result.hand_landmarks
    handedness_list = detection_result.handedness
    
    annotated_image = np.copy(rgb_image)
    h, w, c = annotated_image.shape
    
    for hand_landmarks, handedness in zip(hand_landmarks_list, handedness_list):
        
        for landmark in hand_landmarks:
            x = int(landmark.x * w)
            y = int(landmark.y * h)
            cv2.circle(annotated_image, (x, y), 3, (0, 255, 0), -1)
        
        for connection in HAND_CONNECTIONS:
            start_idx = connection[0]
            end_idx = connection[1]
            
            start_point = hand_landmarks[start_idx]
            end_point = hand_landmarks[end_idx]
            
            x1 = int(start_point.x * w)
            y1 = int(start_point.y * h)
            x2 = int(end_point.x * w)
            y2 = int(end_point.y * h)
            
            cv2.line(annotated_image, (x1, y1), (x2, y2), (255, 0, 0), 2)
        
        label = handedness[0].category_name
        cv2.putText(annotated_image, label, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    return annotated_image


def extract_features(landmarks):
    """
    Extract features from hand landmarks for gesture classification.
    Features include normalized coordinates, finger extension ratios, angles, and fingertip distances.
    Inputs:
        landmarks: List of hand landmarks
    Returns:
        features: Array of extracted features
    """
    features = []
    
    # Normalize relative to wrist (landmark 0)
    wrist_x = landmarks[0].x
    wrist_y = landmarks[0].y
    wrist_z = landmarks[0].z
    
    for lm in landmarks:
        features.extend([
            lm.x - wrist_x,
            lm.y - wrist_y,
            lm.z - wrist_z
        ])
    
    # Finger extension ratios
    finger_tips = [4, 8, 12, 16, 20]
    finger_pips = [3, 7, 11, 15, 19]
    finger_mcps = [2, 6, 10, 14, 18]
    
    for tip, pip, mcp in zip(finger_tips, finger_pips, finger_mcps):
        mcp_tip_dist = math.sqrt(
            (landmarks[tip].x - landmarks[mcp].x)**2 +
            (landmarks[tip].y - landmarks[mcp].y)**2 +
            (landmarks[tip].z - landmarks[mcp].z)**2
        )
        
        mcp_pip_dist = math.sqrt(
            (landmarks[pip].x - landmarks[mcp].x)**2 +
            (landmarks[pip].y - landmarks[mcp].y)**2 +
            (landmarks[pip].z - landmarks[mcp].z)**2
        )
        
        if mcp_pip_dist > 0.001:
            extension_ratio = mcp_tip_dist / mcp_pip_dist
        else:
            extension_ratio = 0.0
        
        features.append(extension_ratio)
    
    # Finger angles
    for tip in finger_tips:
        dy = landmarks[tip].y - landmarks[0].y
        dx = landmarks[tip].x - landmarks[0].x
        angle = math.atan2(dx, dy)
        features.append(angle)
    
    # Fingertip distances
    for i in range(len(finger_tips)):
        for j in range(i+1, len(finger_tips)):
            tip_i = finger_tips[i]
            tip_j = finger_tips[j]
            dist = math.sqrt(
                (landmarks[tip_i].x - landmarks[tip_j].x)**2 +
                (landmarks[tip_i].y - landmarks[tip_j].y)**2 +
                (landmarks[tip_i].z - landmarks[tip_j].z)**2
            )
            features.append(dist)
    
    return np.array(features, dtype=np.float32)


def main():
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Cannot open camera")
        return
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    last_visualize = time.time()
    visualization = False
    
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
    
    print("Loading trained classifier...")
    try:
        with open(CLASSIFIER_PATH, 'rb') as f:
            classifier = pickle.load(f)
        with open(SCALER_PATH, 'rb') as f:
            scaler = pickle.load(f)
        print(f"Classifier classes: {classifier.classes_}")
    except Exception as e:
        print(f"Error loading classifier: {e}")
        return
    
    print("Press ESC to exit")
    print("\nPress 'a' to analyze trajectory, or continue...")
    
    motion_gate = MotionGate(
    buffer_size=20,
    velocity_threshold=0.16,     # Adjust if needed
    acceleration_threshold=0.6   # Adjust if needed
)
    dynamic_detector = DynamicGestureDetector()
    
    frame_count = 0
    fps_time = time.time()
    inference_times = deque(maxlen=30)
    
    confidence_threshold = 0.65
    
    while True:
        ret, frame = cap.read()
        
        if not ret:
            break
        
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        
        start_time = time.perf_counter()
        detection_result = detector.detect(mp_image)
        end_time = time.perf_counter()
        
        inference_ms = (end_time - start_time) * 1000
        inference_times.append(inference_ms)
        avg_inference_ms = np.mean(inference_times)
        
        annotated_frame = draw_landmarks_on_image(rgb_frame, detection_result)
        annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR)
        
        h, w, _ = annotated_frame.shape
        
        predicted_gesture = "no_gesture"
        confidence = 0.0
        motion_type = "unknown"
        
        if len(detection_result.hand_landmarks) > 0:
            landmarks = detection_result.hand_landmarks[0]
            
            # Update motion gate
            motion_gate.update(landmarks)
            motion_type = motion_gate.get_motion_type()

            # # Static path: MLP classification
            if motion_type == 'static':
                features = extract_features(landmarks).reshape(1, -1)
                # print(features)
                features_scaled = scaler.transform(features)
                
                prediction = classifier.predict(features_scaled)[0]
                confidence_values = classifier.predict_proba(features_scaled)[0]
                max_confidence = np.max(confidence_values)
                
                if max_confidence > confidence_threshold:
                    predicted_gesture = prediction
                    confidence = max_confidence
                else:
                    predicted_gesture = "no_gesture"
                    confidence = 1.0 - max_confidence
            
            # Dynamic path: trajectory analysis
            elif motion_type == 'dynamic':
                trajectory_by_method = {
                    'wrist': motion_gate.get_trajectory(method='wrist'),
                    'center': motion_gate.get_trajectory(method='center'),
                    'fingertips': motion_gate.get_trajectory(method='fingertips')
                }
                best_gesture, best_confidence = dynamic_detector.detect_best_match(trajectory_by_method)
                if best_confidence > 0.85:  # Confidence threshold
                    predicted_gesture = best_gesture
                    confidence = best_confidence
                    if time.time() - last_visualize > 5:
                        visualization = True
                        last_visualize = time.time()
                else:
                    predicted_gesture = f"no_gesture but almost {best_gesture}"
               
        
        # Draw status
        y_offset = 30
        cv2.putText(annotated_frame, f"Motion: {motion_type}", (10, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        y_offset += 30
        gesture_color = (0, 255, 0) if predicted_gesture != "no_gesture" else (0, 255, 255)
        cv2.putText(annotated_frame, f"Gesture: {predicted_gesture}", (10, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, gesture_color, 2)
        
        y_offset += 30
        cv2.putText(annotated_frame, f"Confidence: {confidence:.2f}", (10, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # FPS and inference time
        frame_count += 1
        elapsed_time = time.time() - fps_time
        if elapsed_time > 1.0:
            fps = frame_count / elapsed_time
            fps_text = f"FPS: {fps:.1f}"
            cv2.putText(annotated_frame, fps_text, (10, h - 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            frame_count = 0
            fps_time = time.time()
        
        inference_text = f"Inference: {avg_inference_ms:.1f}ms"
        cv2.putText(annotated_frame, inference_text, (10, h - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        cv2.imshow('Gesture Recognition', annotated_frame)

        # visualization
        # if visualization:
        #     dynamic_detector.visualize_templates(trajectory_by_method, normalize=True)
        #     visualization = False
        
        if cv2.waitKey(1) & 0xFF == 27:
            print("\nExit")
            break
    
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()