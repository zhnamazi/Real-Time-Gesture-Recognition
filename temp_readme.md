# Data Generation for Liveness Model

This repository contains scripts used to generate and prepare datasets for training a liveness detection model. The tools cover preprocessing steps such as face cropping, frame extraction, dataset analysis, and CSV generation.

---

## Repository Structure

* **crop_faces.py**
  Crops detected faces and converts them to the largest possible square.

* **crop_square.py**
  Crops the largest square possible from the center of an image or from all images in a folder.

* **dataset_distribution.py**
  Calculates and displays the distribution of class labels in a dataset CSV.

* **extract_frames.py**
  Extracts frames from videos and save them as images.

* **face_detection.py**
  Detects faces in images using the detection model.

* **make_csv.py**
  Creates CSV files from image directories with paths and labels.

* **manual_crop.py**
  Allows manual face cropping for cases where automatic detection fails

* **merge_csv.py**
  Merges multiple CSV files into one.

* **rppg_generator.py**
  Extracts rPPG (remote photoplethysmography) signals from input videos for physiological feature analysis

* **test_fft.py**
  Tests FFT generation for a given image input to verify frequency domain representation.

* **test_lbp.py**
  Tests LBP generation on a single input image for appearance feature verification.

* **visualize_depth_mask.py**
  Visualizes 3D facial mesh depth maps generated using MediaPipe.