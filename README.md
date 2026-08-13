# Real-Time Gesture Recognition with Raspberry Pi

This project implements a liveness detection system using a ResNet-18 backbone and combines RGB images, Local Binary Patterns (LBP), and frequency-domain (FFT) maps as input features. The model is trained to classify whether a face is real or fake (e.g., spoofed using photos or videos).

---

## Features

* Input: RGB + LBP + FFT (5 channels)
* Backbone: Pretrained ResNet-18
* Classifier: Fully connected layers with dropout
* Loss: CrossEntropyLoss
* Optimizer: Adam
* Logging: WandB for experiment tracking
* Training supports:

  * Custom batch sizes
  * Train/Validation splits
  * Checkpoint saving
  * Prediction logging

---

## Project Structure

```
Real-Time-Gesture-Recognition/
│
├─ data_generation/        # 
everythin contains in data_generation
├─ src/                    # LivenessModel definition and architecture
...
├─ training/               # 
...
└─ README.md
```

---

## Setup

1. **Clone the repository**:

```bash
git clone git@gitlab.veerasense.local:omid/ai-modules/new-format/liveness/liveness-model-train.git
cd liveness-model-train
```

2. **Install dependencies**:

```bash
pip install -r requirements.txt
```

3. **Prepare your dataset CSV**:

CSV should contain two columns:

| frame_path          | label |
| ------------------- | ----- |
| /path/to/image1.jpg | real  |
| /path/to/image2.jpg | fake  |

> The label can be `"real"` or `"fake"`.

---

## Training

Run the training script:

```bash
python main.py
```

Training configuration is loaded from `config.yaml`:

* `data`: CSV path, image size, batch size, train/val split, number of workers
* `model`: input channels, feature dimension, number of classes, dropout
* `training`: learning rate, epochs, pretrained model path
* `logging`: log and checkpoint paths, iteration intervals

The trainer automatically logs metrics and predictions to **WandB** if configured.

---

## Dataset

The project uses a custom dataset loader that:

* Loads RGB images
* Computes LBP maps
* Computes FFT maps
* Combines them into a tensor of shape `(C, H, W)` where `C=5` (RGB + LBP + FFT)
* Normalizes each channel to `[0,1]`

---

## Inference

To run inference on a new image:

```python
from test import LivenessInference
import cv2

inference = LivenessInference('path/to/model.pt', device='cpu')
image = cv2.imread("path/to/image.jpg")
output = inference.predict(image)
print(output)
```

---

## Logging and Visualization

* Training logs are saved to `config['logging']['log_path']`
* Model checkpoints are saved to `config['logging']['snapshot_path']`
* WandB can track:

  * Training/validation loss and accuracy
  * Sample predictions