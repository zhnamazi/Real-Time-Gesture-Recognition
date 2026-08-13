"""
This script is used to convert the HaGRID dataset's JSON annotations into a CSV format suitable for training machine learning models.
It extracts enhanced features from the hand landmarks, including normalized coordinates, finger extension ratios, angles, and fingertip distances.
"""

import json
import csv
import math

JSON_FILE = "./datasets/HaGRID/dislike annotation/dislike.json"  # HaGRID JSON file
OUTPUT_CSV = "./datasets/hand_gestures_dataset_v5.csv" # Output CSV file
FILTER_CLASSES = ["like", "dislike"]  # Only these classes

def extract_enhanced_features_from_landmarks(landmarks):
    """
    Convert 21 landmarks to 83 enhanced features
    landmarks: list of 21 (x, y) tuples

    Inputs:
        - landmarks: list of 21 (x, y) tuples representing hand landmarks
    
    Returns:
        - features: list of 83 features including normalized coordinates, finger extension ratios, angles, and fingertip distances
    """
    
    features = []
    
    # Normalize coordinates relative to wrist (landmark 0)
    wrist_x = landmarks[0][0]
    wrist_y = landmarks[0][1]
    
    # Calculate scaling factor: distance from wrist to middle finger tip (landmark 12)
    scale_dist = math.sqrt(
        (landmarks[12][0] - wrist_x)**2 +
        (landmarks[12][1] - wrist_y)**2
    )
    
    if scale_dist < 0.001:
        scale_dist = 1.0
    
    # Normalize: subtract wrist and divide by scale distance
    normalized_landmarks = []
    for x, y in landmarks:
        norm_x = (x - wrist_x) / scale_dist
        norm_y = (y - wrist_y) / scale_dist

        # Add dummy z coordinate (HaGRID only has x,y)
        normalized_landmarks.append((norm_x, norm_y, 0.0))
    
    # Add normalized coordinates (63 features)
    for lm in normalized_landmarks:
        features.extend([lm[0], lm[1], lm[2]])
    
    # Finger extension ratios (5 features)
    finger_tips = [4, 8, 12, 16, 20]
    finger_pips = [3, 7, 11, 15, 19]
    finger_mcps = [2, 6, 10, 14, 18]
    
    for tip, pip, mcp in zip(finger_tips, finger_pips, finger_mcps):
        mcp_tip_dist = math.sqrt(
            (normalized_landmarks[tip][0] - normalized_landmarks[mcp][0])**2 +
            (normalized_landmarks[tip][1] - normalized_landmarks[mcp][1])**2 +
            (normalized_landmarks[tip][2] - normalized_landmarks[mcp][2])**2
        )
        
        mcp_pip_dist = math.sqrt(
            (normalized_landmarks[pip][0] - normalized_landmarks[mcp][0])**2 +
            (normalized_landmarks[pip][1] - normalized_landmarks[mcp][1])**2 +
            (normalized_landmarks[pip][2] - normalized_landmarks[mcp][2])**2
        )
        
        if mcp_pip_dist > 0.001:
            extension_ratio = mcp_tip_dist / mcp_pip_dist
        else:
            extension_ratio = 0.0
        
        features.append(extension_ratio)
    
    # Finger angles (5 features)
    for tip in finger_tips:
        dy = normalized_landmarks[tip][1] - normalized_landmarks[0][1]
        dx = normalized_landmarks[tip][0] - normalized_landmarks[0][0]
        angle = math.atan2(dx, dy)
        features.append(angle)
    
    # Fingertip distances (10 features)
    for i in range(len(finger_tips)):
        for j in range(i+1, len(finger_tips)):
            tip_i = finger_tips[i]
            tip_j = finger_tips[j]
            dist = math.sqrt(
                (normalized_landmarks[tip_i][0] - normalized_landmarks[tip_j][0])**2 +
                (normalized_landmarks[tip_i][1] - normalized_landmarks[tip_j][1])**2 +
                (normalized_landmarks[tip_i][2] - normalized_landmarks[tip_j][2])**2
            )
            features.append(dist)
    
    return features

def main():
    print(f"Loading HaGRID annotations from {JSON_FILE}...")
    
    try:
        with open(JSON_FILE, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: File {JSON_FILE} not found")
        return
    except json.JSONDecodeError:
        print(f"Error: {JSON_FILE} is not valid JSON")
        return
    
    print(f"Total images in annotation file: {len(data)}")
    
    # Create CSV header
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
    for i in range(5):
        for j in range(i+1, 5):
            header.append(f'tip_dist_{i}_{j}')
    
    # Write header
    with open(OUTPUT_CSV, mode='a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
    
    sample_count = 0
    skipped_count = 0
    error_count = 0
    
    # Process each image entry
    for image_id, image_data in data.items():
        try:
            # Check if image has landmarks
            if 'landmarks' not in image_data or not image_data['landmarks']:
                skipped_count += 1
                continue
            
            # Check labels
            if 'labels' not in image_data or not image_data['labels']:
                skipped_count += 1
                continue
            
            # Process each hand in the image
            for hand_idx, landmarks_list in enumerate(image_data['landmarks']):
                
                # Check if we have a label for this hand
                if hand_idx >= len(image_data['labels']):
                    break
                
                label = image_data['labels'][hand_idx]
                
                # Only process like and dislike
                if label not in FILTER_CLASSES:
                    skipped_count += 1
                    continue
                
                # Check if landmarks are valid
                if not landmarks_list or len(landmarks_list) != 21:
                    skipped_count += 1
                    continue
                
                # Convert landmarks to list of tuples
                landmarks = []
                for lm in landmarks_list:
                    if isinstance(lm, (list, tuple)) and len(lm) >= 2:
                        landmarks.append((float(lm[0]), float(lm[1])))
                    else:
                        raise ValueError("Invalid landmark format")
                
                if len(landmarks) != 21:
                    skipped_count += 1
                    continue
                
                # Extract enhanced features
                features = extract_enhanced_features_from_landmarks(landmarks)
                
                # Write to CSV
                row = [label] + features
                with open(OUTPUT_CSV, mode='a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(row)
                
                sample_count += 1
                
                if sample_count % 1000 == 0:
                    print(f"  Processed {sample_count} samples...")
        
        except Exception as e:
            error_count += 1
            if error_count <= 5:  # Print first 5 errors only
                print(f"  Warning (image {image_id}): {str(e)}")
            continue
    
    print(f"\nDone!")
    print(f"Samples processed: {sample_count}")
    print(f"Samples skipped: {skipped_count}")
    print(f"Errors encountered: {error_count}")
    
    # Count by class
    import pandas as pd
    df = pd.read_csv(OUTPUT_CSV)
    print(f"\nSamples per class:")
    print(df['label'].value_counts())

if __name__ == "__main__":
    main()