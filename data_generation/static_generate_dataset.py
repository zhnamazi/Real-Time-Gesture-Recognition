import math

import cv2
import csv
import os
import mediapipe as mp

# Configuration
MODEL_PATH = "./src/hand_landmarker.task" 
CSV_FILE = "./dataset/hand_gestures_dataset_v6.csv"
TARGET_SAMPLES = 1500
SAMPLE_RATE = 5  # Frames per second. default fps is 30

def extract_enhanced_features(landmarks):
    """
    Extract richer features from landmarks:
    - Normalized coordinates (63 features)
    - Finger extension ratios (5 features)
    - Finger angles (5 features)
    - Fingertip distances (10 features)
    Total: 83 features
    """
    
    features = []
    
    # Normalize coordinates
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
    
    return features

# Get class label from user
class_name = input("Enter the class label for this recording session (e.g., like, dislike, unknown): ").strip()
if not class_name:
    class_name = "unknown"

# Prepare CSV file for saving landmarks
# x, y, z for each of the 21 landmarks
file_exists = os.path.exists(CSV_FILE)

# Create header for CSV: label, x0, y0, z0, x1, y1, z1, ..., x20, y20, z20
header = ['label']
# Coordinate features (63)
for i in range(21):
    header.extend([f'x{i}', f'y{i}', f'z{i}'])
# Extension ratios (5)
for i in range(5):
    header.append(f'extension{i}')
# Angles (5)
for i in range(5):
    header.append(f'angle{i}')
# Tip distances (10)
idx = 0
for i in range(5):
    for j in range(i+1, 5):
        header.append(f'tip_dist_{i}_{j}')

# If the file doesn't exist, write the header
if not file_exists:
    with open(CSV_FILE, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)


# Initialize MediaPipe Hand Landmarker
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=VisionRunningMode.VIDEO,
    num_hands=1
)

landmarker = HandLandmarker.create_from_options(options)


# Loop to capture video and record landmarks
cap = cv2.VideoCapture(0)
is_recording = False  # Default state is paused
with open(CSV_FILE, mode='r') as f:
    reader = csv.reader(f)
    header = next(reader, None)  # Skip the header
    sample_count = sum(1 for row in reader if row and row[0] == class_name)
frame_timestamp = 0

print("\n--- Key Instructions ---")
print(" [SPACE] : (Pause / Resume)")
print(" [Q]     : Exit the program")
print("----------------------\n")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)  # Mirror Mode
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

    frame_timestamp += 1
    result = landmarker.detect_for_video(mp_image, frame_timestamp)

    # Check if any hands are detected
    if result.hand_landmarks:
        landmarks = result.hand_landmarks[0]

        is_sample_frame = is_recording and (frame_timestamp % SAMPLE_RATE == 0)
        point_color = (0, 255, 0) if is_sample_frame else (255, 100, 0)
        point_radius = 6 if is_sample_frame else 3
        
        # Draw landmarks on the frame
        for lm in landmarks:
            h, w, _ = frame.shape
            cx, cy = int(lm.x * w), int(lm.y * h)
            cv2.circle(frame, (cx, cy), point_radius, point_color, -1)

        # If we are in recording mode
        if is_sample_frame:
            cv2.putText(frame, "RECORDING SAMPLE!", (10, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            enhanced_features = extract_enhanced_features(landmarks)
            row = [class_name] + enhanced_features

            # Append the row to the CSV file
            with open(CSV_FILE, mode='a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(row)

            sample_count += 1

            # Check if we reached the target number of samples
            if sample_count >= TARGET_SAMPLES:
                print(f"\n✅ Target of {TARGET_SAMPLES} samples reached for class '{class_name}'!")
                break

    # Display recording status and sample count
    status_text = "RECORDING" if is_recording else "PAUSED (Press SPACE)"
    status_color = (0, 0, 255) if is_recording else (0, 255, 255)

    # Record the status and sample count on the frame
    cv2.putText(frame, f"Status: {status_text}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
    
    # Class name and sample count
    cv2.putText(frame, f"Class: {class_name}", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    cv2.putText(frame, f"Samples: {sample_count} / {TARGET_SAMPLES}", (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    cv2.imshow("Dataset Collector", frame)


    # Manage key presses
    key = cv2.waitKey(1) & 0xFF
    if key == ord(' '):  # Space key to Pause/Resume
        is_recording = not is_recording
    elif key == ord('q'):  # Q key to exit
        break

cap.release()
cv2.destroyAllWindows()
landmarker.close()
print(f"\nDone! Saved {sample_count} samples for '{class_name}' into '{CSV_FILE}'.")