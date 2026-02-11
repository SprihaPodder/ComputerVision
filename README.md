# Computer Vision Portfolio

This repository contains a collection of real-time computer vision applications built using OpenCV, MediaPipe, and modern human–computer interaction techniques. The projects focus on gesture recognition, object tracking, interactive vision systems, and real-time augmented interfaces.

These applications are designed to run with a webcam and emphasize low-latency real-time processing suitable for research prototypes and interactive systems.

---

## Overview

This portfolio demonstrates:

- Real-time color segmentation and centroid tracking
- Gesture-based human–computer interaction
- Interactive vision games and drawing systems
- Motion detection and surveillance pipelines
- Augmented reality visual effects
- Virtual musical instruments
- Gesture-controlled dataset navigation
- Virtual mouse and game control
- YOLO real-time object detection

---

## Tech Stack

### Languages
- C++
- Python

### Libraries and Frameworks
- OpenCV
- MediaPipe
- NumPy
- PyAutoGUI
- YOLO (You Only Look Once)

### Core Concepts
- HSV color segmentation
- Image moments and centroid computation
- Background subtraction
- Gesture recognition
- Collision detection
- Coordinate mapping
- Real-time UI composition
- Motion tracking
- Augmented reality overlays

---

## Projects

### App 1: Red Pixel Detection (C++ / OpenCV)

Real-time detection of red regions using HSV color thresholding. Dual HSV ranges handle hue wrap-around. Binary masking and morphological filtering isolate red pixels.

**Key techniques**
- HSV color segmentation
- Thresholding
- Morphological noise removal

**Applications**
Color-based object tracking and gesture detection.

---

### App 2: Centroid Tracking of Red Pixels (C++ / OpenCV)

Extends red detection by computing the barycenter of detected pixels using image moments. The centroid is tracked and visualized in real time.

**Key techniques**
- Image moments
- Centroid computation
- Real-time motion tracking

---

### App 3: Gesture Balloon Pop Game (Python / OpenCV / MediaPipe)

A gesture-controlled interactive game where fingertip tracking is used to pop balloons. Includes scoring, animations, sound effects, and a leaderboard.

**Key techniques**
- Hand landmark detection
- Collision detection
- Game state management
- Real-time gesture interaction

---

### App 4: Virtual Remote Painter (C++ / OpenCV)

Tracks centroid motion to simulate a virtual paintbrush. Color palettes and a clear button enable gesture-based drawing on a live video canvas.

**Key techniques**
- Centroid motion tracking
- Interactive UI overlays
- Path rendering

---

### App 5: Motion Detection and Intrusion Alarm (C++ / OpenCV)

Adaptive background modeling detects large motion events. Intrusions trigger alarms and automatically save timestamped snapshots.

**Key techniques**
- Background subtraction
- Contour filtering
- Real-time alert systems

---

### App 6: Invisibility Cloak Effect (C++ / OpenCV)

Creates an augmented reality invisibility illusion by replacing cloak-colored pixels with a stored background.

**Key techniques**
- Color masking
- Background replacement
- Image compositing

---

### App 7: Virtual Piano (Python / OpenCV / MediaPipe)

A gesture-controlled musical interface where fingertips trigger notes mapped to on-screen piano keys.

**Key techniques**
- Region-based gesture detection
- Audio playback threading
- Interactive UI rendering

---

### App 8: Gesture Dataset Viewer (Python / OpenCV / MediaPipe)

Allows hands-free browsing of image datasets using swipe navigation and pinch-to-zoom gestures.

**Key techniques**
- Gesture classification
- Coordinate transformation
- Dynamic image scaling
- Multi-panel UI composition

---

### App 9: Gesture Mouse and Game Controller (Python / OpenCV / MediaPipe)

Transforms hand gestures into mouse movement, clicks, and drag operations for controlling web games.

**Key techniques**
- Cursor coordinate mapping
- Gesture threshold detection
- Motion smoothing filters

---

### YOLO Object Detection

Implements YOLO for high-speed object detection on images and live webcam streams. Bounding boxes and labels are drawn in real time.

**Key techniques**
- Deep neural network inference
- Real-time object localization
- Video frame processing

---

## Mathematical Foundations

Several core mathematical models are shared across projects.

### Coordinate Mapping

Normalized landmark coordinates are mapped to pixel space:
x = x_norm * W
y = y_norm * H

### Centroid Computation (Image Moments)

x_c = sum(x * M) / sum(M)
y_c = sum(y * M) / sum(M)

### Euclidean Distance for Gesture Detection

d = sqrt((x1 - x2)^2 + (y1 - y2)^2)

### Adaptive Background Update

B_t = (1 - alpha) * B_(t-1) + alpha * I_t

### Binary Thresholding

Thresholding isolates significant regions for tracking and interaction.

---

## Installation

### Python Dependencies

pip install opencv-python mediapipe numpy pyautogui

### C++ Requirements

- OpenCV installed and configured
- C++ compiler with OpenCV linkage

---

## How to Run

1. Clone the repository.
2. Navigate to the desired project folder.
3. Ensure webcam access is enabled.
4. Run the Python script or compile and execute the C++ program.

Each project folder contains independent source code and assets.

---

## Applications

- Gesture-based interfaces
- Smart surveillance systems
- Interactive gaming
- Augmented reality visualization
- Educational demonstrations
- Human–computer interaction research

---

## Future Enhancements

- Multi-hand gesture support
- GPU acceleration
- Deep learning gesture classification
- AR/VR integration
- Mobile deployment

---

## Author

Spriha Podder  
Computer Vision Portfolio
