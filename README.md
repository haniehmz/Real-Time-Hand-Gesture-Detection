# Real Time Hand Gesture Detection Using YOLOv8

A real-time hand gesture detection project built with **YOLOv8, OpenCV, and MediaPipe**. The project uses a custom dataset created from personal hand gesture videos to detect six classes: numbers one through five and the OK gesture.

Unlike projects that only run pretrained object detection models, this project covers the complete custom object detection workflow, including video frame extraction, automatic bounding-box annotation, model training, and real-time webcam inference.

## Overview

Hand gesture recognition is an important computer vision application that enables intuitive interaction between humans and computer systems.

This project implements a custom object detection pipeline that identifies six hand gestures:

- One finger
- Two fingers
- Three fingers
- Four fingers
- Five fingers
- OK gesture (👌🏻)

The dataset was created by recording hand gesture videos and extracting individual frames. MediaPipe was then used to detect hand landmarks and automatically generate bounding-box annotations for YOLO training.

A YOLOv8 Nano model was fine-tuned on the resulting dataset and used for real-time inference through a webcam.

## Key Features

- Custom dataset collected from hand gesture videos
- Automated video-to-image frame extraction
- Automatic bounding-box annotation using MediaPipe
- Six custom gesture classes
- YOLOv8 Nano model training
- Custom object detection using Ultralytics YOLO
- Real-time webcam detection
- Bounding-box and confidence-score visualization
- End-to-end computer vision workflow

## Gesture Classes

The model is designed to detect the following six classes:

| Class ID | Class Name | Description |
|---:|---|---|
| 0 | `one` | Hand gesture representing one |
| 1 | `two` | Hand gesture representing two |
| 2 | `three` | Hand gesture representing three |
| 3 | `four` | Hand gesture representing four |
| 4 | `five` | Hand gesture representing five |
| 5 | `ok` | OK hand gesture |

The class IDs correspond to the order defined in the dataset preparation code.

## Custom Dataset Creation

One of the main contributions of this project is the creation of a custom dataset instead of relying entirely on an existing labeled dataset.

### 1. Video Collection

Videos of the six hand gestures were recorded and organized into class-specific folders.

```text
dataset/
└── videos/
    ├── one/
    ├── two/
    ├── three/
    ├── four/
    ├── five/
    └── ok/
```

Each directory contains videos representing its corresponding gesture.

### 2. Frame Extraction

OpenCV was used to extract frames from the recorded videos.

The extraction process samples one frame for every eight frames in the original video.

```python
FRAME_STEP = 8
```

Extracted images are organized by class:

```text
dataset/
└── images/
    ├── one/
    ├── two/
    ├── three/
    ├── four/
    ├── five/
    └── ok/
```

This converts the original video recordings into an image dataset suitable for object detection training.

### 3. Automatic Annotation with MediaPipe

MediaPipe Hands was used to detect hand landmarks in each image.

The minimum and maximum landmark coordinates define an initial hand bounding box. Padding is then added around the box to include additional context.

```python
PADDING = 0.2
```

The bounding box is converted into normalized YOLO annotation format:

```text
class_id x_center y_center width height
```

Each annotated image receives a corresponding `.txt` file.

This automated labeling process reduces manual annotation effort. Images in which MediaPipe does not detect a hand are skipped.

### Dataset Annotation Format

YOLO annotation coordinates are normalized relative to image width and height.

| Field | Description |
|---|---|
| `class_id` | Integer identifying the gesture class |
| `x_center` | Normalized bounding-box center x-coordinate |
| `y_center` | Normalized bounding-box center y-coordinate |
| `width` | Normalized bounding-box width |
| `height` | Normalized bounding-box height |

The class ID identifies the gesture associated with the image's source video folder.

## Model Architecture

The project uses **YOLOv8 Nano (`yolov8n.pt`)**, a lightweight object detection model provided by Ultralytics.

The model is initialized with pretrained weights and trained on the custom six-class dataset.

### Training Configuration

| Parameter | Value |
|---|---|
| Base model | YOLOv8 Nano |
| Initial weights | `yolov8n.pt` |
| Number of classes | 6 |
| Training epochs | 40 |
| Image size | 640 × 640 |
| Batch size | 16 |
| Device | CPU |
| Workers | 0 |

The pretrained model provides an initialization that is fine-tuned for the custom hand gesture detection task.

## Training Workflow

```text
Hand Gesture Videos
        │
        ▼
Video Frame Extraction
        │
        ▼
Custom Image Dataset
        │
        ▼
MediaPipe Hand Landmark Detection
        │
        ▼
Automatic Bounding-Box Annotation
        │
        ▼
YOLO Dataset Preparation
        │
        ▼
YOLOv8 Nano Training
        │
        ▼
Trained Model Weights
        │
        ▼
Real-Time Webcam Detection
```

## Real-Time Webcam Detection

After training, the best model weights are loaded for real-time inference.

The webcam provides live video frames, and YOLOv8 predicts the locations and classes of detected gestures.

The confidence threshold is set to `0.5`, meaning detections below this threshold are filtered out.

The model can detect the six trained classes when they are visible and sufficiently similar to the examples represented in the training data.


## Installation

Python 3.10 or 3.11 is a reasonable starting point, but the exact compatible versions depend on the installed Ultralytics, PyTorch, and MediaPipe releases.

Install the required packages:

```bash
pip install ultralytics opencv-python mediapipe
```

The notebook also uses standard Python libraries such as `os` and `math`.

## Output example

Some examples of test output with webcam:

<img width="600" height="450" alt="out1" src="https://github.com/user-attachments/assets/b49a672f-698f-4ffb-b087-60b79461f276" />

<img width="600" height="450" alt="out2" src="https://github.com/user-attachments/assets/51818cc7-589c-4cea-9d09-1f08b6c0a854" />
