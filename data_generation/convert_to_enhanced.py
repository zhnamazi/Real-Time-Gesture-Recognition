#!/usr/bin/env python3

import pandas as pd
import numpy as np
import math
import csv

OLD_CSV = "../dataset/hand_gestures_dataset_v3.csv"  # Your old file with 63 features
NEW_CSV = "../dataset/hand_gestures_dataset_v4.csv"  # New file with 83 features

print(f"Reading old dataset from {OLD_CSV}...")
df = pd.read_csv(OLD_CSV)

print(f"Dataset shape: {df.shape}")
print(f"Classes: {df['label'].unique()}")

# Create new CSV with enhanced features
header = ['label']
# Coordinate features (63) - same as before
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
with open(NEW_CSV, mode='w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(header)

print(f"Processing {len(df)} samples...")

# Process each row
for idx, row in df.iterrows():
    class_name = row['label']
    
    # Extract coordinates (x0, y0, z0, x1, y1, z1, ...)
    coords = []
    coords_list = []
    for i in range(21):
        x = row[f'x{i}']
        y = row[f'y{i}']
        z = row[f'z{i}']
        coords.append((x, y, z))
        coords_list.extend([x, y, z])
    
    # Calculate enhanced features
    # features = list(coords)  # Flatten: 63 values
    features = coords_list
    
    # Finger indices
    finger_tips = [4, 8, 12, 16, 20]
    finger_pips = [3, 7, 11, 15, 19]
    finger_mcps = [2, 6, 10, 14, 18]
    
    # 1. Extension ratios
    for tip, pip, mcp in zip(finger_tips, finger_pips, finger_mcps):
        mcp_tip_dist = math.sqrt(
            (coords[tip][0] - coords[mcp][0])**2 +
            (coords[tip][1] - coords[mcp][1])**2 +
            (coords[tip][2] - coords[mcp][2])**2
        )
        
        mcp_pip_dist = math.sqrt(
            (coords[pip][0] - coords[mcp][0])**2 +
            (coords[pip][1] - coords[mcp][1])**2 +
            (coords[pip][2] - coords[mcp][2])**2
        )
        
        if mcp_pip_dist > 0.001:
            extension_ratio = mcp_tip_dist / mcp_pip_dist
        else:
            extension_ratio = 0.0
        
        features.append(extension_ratio)
    
    # 2. Finger angles
    for tip in finger_tips:
        dy = coords[tip][1] - coords[0][1]
        dx = coords[tip][0] - coords[0][0]
        angle = math.atan2(dx, dy)
        features.append(angle)
    
    # 3. Fingertip distances
    for i in range(len(finger_tips)):
        for j in range(i+1, len(finger_tips)):
            tip_i = finger_tips[i]
            tip_j = finger_tips[j]
            dist = math.sqrt(
                (coords[tip_i][0] - coords[tip_j][0])**2 +
                (coords[tip_i][1] - coords[tip_j][1])**2 +
                (coords[tip_i][2] - coords[tip_j][2])**2
            )
            features.append(dist)
    
    # Write to new CSV
    new_row = [class_name] + features
    with open(NEW_CSV, mode='a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(new_row)
    
    if (idx + 1) % 100 == 0:
        print(f"  Processed {idx + 1}/{len(df)} samples...")

print(f"\nDone! New dataset saved to {NEW_CSV}")
print(f"Total samples: {len(df)}")

# Verify
df_new = pd.read_csv(NEW_CSV)
print(f"New dataset shape: {df_new.shape}")
print(f"Classes: {df_new['label'].unique()}")