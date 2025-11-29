<div align="center">

# 🏍️ Helmetless Rider Detection System

### AI-Powered Traffic Safety Enforcement using Deep Learning

[![Python](https://img.shields.io/badge/Python-3.7+-blue.svg?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-1.15-orange.svg?style=for-the-badge&logo=tensorflow&logoColor=white)](https://tensorflow.org)
[![Keras](https://img.shields.io/badge/Keras-2.3.1-red.svg?style=for-the-badge&logo=keras&logoColor=white)](https://keras.io)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.1.2-green.svg?style=for-the-badge&logo=opencv&logoColor=white)](https://opencv.org)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](LICENSE)

<p align="center">
  <strong>Automated detection of motorcycle riders without helmets and extraction of their license plates for traffic law enforcement.</strong>
</p>

---

[Features](#-features) •
[Architecture](#-architecture) •
[Installation](#-installation) •
[Usage](#-usage) •
[Training](#-training) •
[Configuration](#-configuration) •
[Contributing](#-contributing)

</div>

---

## 📋 Overview

With the exponential increase of traffic in India, the number of accidents is also increasing exponentially. Most of the reported deaths due to accidents involving two-wheelers are caused due to not wearing a helmet. There is thus a need for better implementation of safety regulation.

This project helps traffic police in better law enforcement through automation in the detection of law-breakers - riders without helmets. The system:

- 🎯 **Classifies two-wheelers** in moving traffic video
- 👤 **Extracts head portions** of riders on two-wheelers  
- 🪖 **Detects helmet presence** (with helmet / without helmet)
- 🔢 **Extracts license plates** when violations are detected
- 📝 **OCR processing** to get registration numbers
- 📊 **Automated reporting** to RTO office

This project eliminates human error and overcomes challenges like heavy traffic or riders who don't stop when called by officers.

---

## ✨ Features

<table>
<tr>
<td>

### 🔍 Detection Capabilities
- **YOLOv3-based object detection** for real-time performance
- **Multi-class detection**: License Plate, Person Bike, With Helmet, Without Helmet
- **Configurable confidence thresholds** for precision tuning
- **Non-Maximum Suppression (NMS)** for accurate bounding boxes

</td>
<td>

### 🎥 Video Processing
- **Real-time video stream processing**
- **Batch frame processing** for efficiency
- **FPS display and monitoring**
- **Support for webcam, video files, and image directories**

</td>
</tr>
<tr>
<td>

### 📍 Object Tracking
- **Centroid-based object tracking**
- **Persistent ID assignment** across frames
- **Disappearance handling** for temporary occlusions
- **Multi-object tracking** support

</td>
<td>

### 🔠 License Plate Recognition
- **Custom ResNet-based OCR model**
- **Image preprocessing** and alignment
- **Adaptive thresholding** techniques
- **Character-level recognition**

</td>
</tr>
</table>

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        HELMETLESS RIDER DETECTION SYSTEM                     │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   ┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────┐ │
│   │   Video      │    │   YOLOv3     │    │   Centroid   │    │  License │ │
│   │   Input      │───▶│   Detector   │───▶│   Tracker    │───▶│  Plate   │ │
│   │              │    │              │    │              │    │  OCR     │ │
│   └──────────────┘    └──────────────┘    └──────────────┘    └──────────┘ │
│         │                    │                    │                  │      │
│         │                    │                    │                  │      │
│         ▼                    ▼                    ▼                  ▼      │
│   ┌──────────────────────────────────────────────────────────────────────┐ │
│   │                        OUTPUT CATEGORIES                              │ │
│   │  • License Plate  • Person Bike  • With Helmet  • Without Helmet     │ │
│   └──────────────────────────────────────────────────────────────────────┘ │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Model Architecture

The system uses **YOLOv3 (You Only Look Once v3)** architecture built on **Darknet-53** backbone:

```
Input (416x416x3)
        │
        ▼
┌───────────────────┐
│   Darknet-53      │  ◄── 53 Convolutional Layers
│   Backbone        │      Residual Connections
└───────────────────┘
        │
        ├─────────┬─────────┐
        ▼         ▼         ▼
┌───────────┐ ┌───────────┐ ┌───────────┐
│  Scale 1  │ │  Scale 2  │ │  Scale 3  │
│  (13x13)  │ │  (26x26)  │ │  (52x52)  │
│  Large    │ │  Medium   │ │  Small    │
│  Objects  │ │  Objects  │ │  Objects  │
└───────────┘ └───────────┘ └───────────┘
        │         │         │
        └─────────┼─────────┘
                  ▼
        ┌───────────────┐
        │  NMS & Output │
        │  4 Classes    │
        └───────────────┘
```

---

## 📁 Project Structure

```
helmetless-rider-detection/
│
├── 📂 libs/                          # Core utility libraries
│   ├── 🐍 object_detector.py         # YOLOv3 Object Detector class
│   ├── 🐍 centroidtracker.py         # Object tracking implementation
│   ├── 🐍 opencv_utils.py            # OpenCV utility functions
│   ├── 🐍 split_data.py              # Dataset splitting utilities
│   └── 🐍 logger.py                  # Logging configuration
│
├── 📂 train_detector/                # YOLOv3 Training module
│   ├── 🐍 train.py                   # Main training script
│   ├── 🐍 yolo.py                    # YOLOv3 model architecture
│   ├── 🐍 generator.py               # Data batch generator
│   ├── 🐍 voc.py                     # VOC annotation parser
│   ├── 🐍 predict.py                 # Inference script
│   ├── 🐍 evaluate.py                # Model evaluation
│   ├── 🐍 gen_anchors.py             # Anchor box generation
│   ├── 🐍 callbacks.py               # Training callbacks
│   ├── 📂 utils/                     # Training utilities
│   │   ├── 🐍 bbox.py                # Bounding box utilities
│   │   ├── 🐍 utils.py               # General utilities
│   │   ├── 🐍 colors.py              # Color mapping
│   │   └── 🐍 image.py               # Image augmentation
│   └── 📄 requirements.txt           # Training dependencies
│
├── 📂 lp_recognizer/                 # License Plate OCR module
│   ├── 🐍 model.py                   # Custom ResNet OCR model
│   └── 🐍 lp_allignment.py           # Plate preprocessing
│
├── 📂 data_generator/                # Dataset generation tools
│   ├── 📂 license_plates/            # LP extraction utilities
│   │   ├── 🐍 license_plate_extractor.py
│   │   └── 🐍 random_image_extractor.py
│   └── 📂 synthetic/                 # Synthetic data generation
│       └── 🐍 OD_datagen.py          # Object detection data generator
│
├── 📂 weights/                       # Model weights & configs
│   ├── 📄 config.json                # Model configuration
│   └── 📄 README.md                  # Weights download link
│
├── 📂 tf_logs/                       # TensorBoard logs
│
├── 🐍 main.py                        # Main application entry point
├── 🐍 YOLOv3_OpenCV.py               # OpenCV-based YOLOv3 implementation
├── 🐍 YOLOv3_TF_old.py               # TensorFlow YOLOv3 implementation
└── 🐍 letterbox.py                   # Image letterboxing utility
```

---

## 🚀 Installation

### Prerequisites

- Python 3.7+
- CUDA 10.0+ (for GPU support)
- cuDNN 7.6+ (for GPU support)

### Step 1: Clone the Repository

```bash
git clone https://github.com/animikhaich/helmetless-rider-detection.git
cd helmetless-rider-detection
```

### Step 2: Create Virtual Environment (Recommended)

```bash
# Using venv
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate     # Windows

# Using conda
conda create -n helmet-detection python=3.7
conda activate helmet-detection
```

### Step 3: Install Dependencies

```bash
# Install training dependencies
pip install -r train_detector/requirements.txt

# Core dependencies
pip install tensorflow-gpu==1.15.0  # or tensorflow==1.15.0 for CPU
pip install keras==2.3.1
pip install opencv-contrib-python==4.1.2.30
pip install numpy scipy tqdm imutils
```

### Step 4: Download Model Weights

Download the pre-trained weights from [Google Drive](https://drive.google.com/drive/folders/1pDChvq_wqWFJnojd0Dtm9drT8tKWDO9H?usp=sharing) and place them in the `weights/` directory.

---

## 💻 Usage

### Quick Start - Object Detection

```python
from libs.object_detector import YOLOObjectDetector
import cv2

# Initialize detector
detector = YOLOObjectDetector(
    config_path='weights/config.json',
    input_img_dims=416,
    obj_thresh=0.85,
    nms_thresh=0.25
)

# Load and process image
image = cv2.imread('path/to/image.jpg')
result_image, boxes, labels = detector.detect(image)

# Display results
cv2.imshow('Detection', result_image)
cv2.waitKey(0)
```

### Video Processing

```python
from libs.object_detector import YOLOObjectDetector
import cv2
import time

# Initialize
detector = YOLOObjectDetector(obj_thresh=0.85, nms_thresh=0.25, input_img_dims=800)
cap = cv2.VideoCapture('path/to/video.mp4')

while True:
    start_time = time.time()
    ret, frame = cap.read()
    
    if not ret:
        break
    
    # Detect objects
    modified_frame, boxes, labels = detector.detect(frame)
    
    # Calculate and display FPS
    fps = 1 / (time.time() - start_time)
    cv2.putText(modified_frame, f"FPS: {fps:.2f}", (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
    cv2.imshow("Stream", modified_frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

### Using OpenCV DNN Backend

```python
from YOLOv3_OpenCV import ObjectTrackerYOLO

# Initialize with OpenCV DNN
tracker = ObjectTrackerYOLO(minThreshold=0.5, minConfidence=0.45)

# Process video
cap = cv2.VideoCapture('path/to/video.mp4')
while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    processed_frame = tracker.per_frame_function(frame)
    cv2.imshow("Detection", processed_frame)
    
    if cv2.waitKey(1) == ord('q'):
        break
```

### Command Line Prediction

```bash
# Run prediction on image
python train_detector/predict.py -c weights/config.json -i path/to/image.jpg -o output/

# Run prediction on video
python train_detector/predict.py -c weights/config.json -i path/to/video.mp4 -o output/

# Run prediction on webcam
python train_detector/predict.py -c weights/config.json -i webcam -o output/
```

---

## 🎓 Training

### Dataset Preparation

1. **Prepare annotations** in Pascal VOC XML format
2. **Organize images** and annotations in separate folders
3. **Update config.json** with paths

### Configuration File Setup

```json
{
  "model": {
    "min_input_size": 416,
    "max_input_size": 416,
    "anchors": [14, 20, 21, 41, 30, 19, 32, 147, 37, 66, 52, 203, 72, 295, 110, 384, 35311, 38369],
    "labels": ["License Plate", "Person Bike", "With Helmet", "Without Helmet"]
  },
  "train": {
    "train_image_folder": "path/to/train/images/",
    "train_annot_folder": "path/to/train/annotations/",
    "batch_size": 8,
    "learning_rate": 0.01,
    "nb_epochs": 100000,
    "warmup_epochs": 3,
    "saved_weights_name": "weights/helmet_lp.h5"
  }
}
```

### Generate Custom Anchors

```bash
cd train_detector
python gen_anchors.py -c config.json
```

### Start Training

```bash
python train_detector/train.py -c weights/config.json
```

### Monitor Training with TensorBoard

```bash
tensorboard --logdir=logs/
```

### Evaluate Model

```bash
python train_detector/evaluate.py -c weights/config.json
```

---

## ⚙️ Configuration

### Detection Parameters

| Parameter | Description | Default | Range |
|-----------|-------------|---------|-------|
| `input_img_dims` | Input image size | 416 | 320-608 (multiples of 32) |
| `obj_thresh` | Object confidence threshold | 0.85 | 0.0-1.0 |
| `nms_thresh` | NMS IoU threshold | 0.25 | 0.0-1.0 |

### Training Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `batch_size` | Training batch size | 8 |
| `learning_rate` | Initial learning rate | 0.01 |
| `warmup_epochs` | Warmup epochs | 3 |
| `ignore_thresh` | IoU threshold for ignoring | 0.5 |

### Detection Classes

| Class ID | Class Name | Description |
|----------|------------|-------------|
| 0 | License Plate | Vehicle license plate |
| 1 | Person Bike | Person on motorcycle |
| 2 | With Helmet | Rider wearing helmet |
| 3 | Without Helmet | Rider not wearing helmet |

---

## 🛠️ Modules

### Object Detector (`libs/object_detector.py`)

```python
class YOLOObjectDetector:
    """
    Generalized YOLOv3 Detector implemented with Keras for TensorFlow 1.x
    
    Features:
    - Automatic GPU memory management
    - Configurable confidence thresholds
    - Batch inference support
    """
    
    def __init__(self, config_path, input_img_dims, obj_thresh, nms_thresh):
        # Initialize detector with configuration
        pass
    
    def detect(self, image):
        # Returns: (annotated_image, bounding_boxes, labels)
        pass
```

### Centroid Tracker (`libs/centroidtracker.py`)

```python
class CentroidTracker:
    """
    Object tracking using centroid-based algorithm
    
    Features:
    - Persistent object IDs across frames
    - Automatic deregistration of disappeared objects
    - Efficient distance-based matching
    """
    
    def __init__(self, maxDisappeared=50):
        pass
    
    def update(self, rects):
        # Returns: dictionary of {object_id: centroid}
        pass
```

### Synthetic Data Generator (`data_generator/synthetic/OD_datagen.py`)

```python
class DataGenerator:
    """
    Generate synthetic training data by combining objects with backgrounds
    
    Features:
    - Random object placement
    - Overlap prevention
    - Automatic VOC XML annotation generation
    """
    
    def __init__(self, bg_root_path, objects_root_paths, ...):
        pass
    
    def main(self, total_combinations=200):
        # Generate synthetic dataset with annotations
        pass
```

---

## 📊 Model Performance

### Detection Metrics

The YOLOv3 model is evaluated using:

- **mAP (mean Average Precision)** across all classes
- **IoU (Intersection over Union)** thresholds: 0.5, 0.75
- **Recall** at different confidence thresholds

### Inference Speed

| Resolution | GPU | FPS |
|------------|-----|-----|
| 416x416 | GTX 1080 Ti | ~45 |
| 608x608 | GTX 1080 Ti | ~25 |
| 416x416 | CPU | ~2-5 |

---

## 🔧 Troubleshooting

### Common Issues

<details>
<summary><b>CUDA/GPU Memory Error</b></summary>

```python
# Enable memory growth
import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)
```
</details>

<details>
<summary><b>Model Loading Error</b></summary>

Ensure the weights file path in `config.json` is correct:
```json
{
  "train": {
    "saved_weights_name": "weights/helmet_lp.h5"
  }
}
```
</details>

<details>
<summary><b>OpenCV Display Issues</b></summary>

For headless servers, use:
```python
import matplotlib
matplotlib.use('Agg')
cv2.imwrite('output.jpg', frame)  # Instead of cv2.imshow()
```
</details>

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📝 TODO

- [ ] Add support for TensorFlow 2.x
- [ ] Implement real-time alert system
- [ ] Add REST API for inference
- [ ] Create Docker container
- [ ] Add mobile deployment support
- [ ] Integrate with traffic management systems

---

## 📚 References

- [YOLOv3: An Incremental Improvement](https://arxiv.org/abs/1804.02767) - Joseph Redmon, Ali Farhadi
- [Darknet Framework](https://pjreddie.com/darknet/yolo/)
- [Keras YOLO Implementation](https://github.com/experiencor/keras-yolo3)

---

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 👤 Author

**Animikh Aich**

- GitHub: [@animikhaich](https://github.com/animikhaich)

---

<div align="center">

### ⭐ Star this repository if you find it helpful!

<p>Made with ❤️ for safer roads</p>

</div>
