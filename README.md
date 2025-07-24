# Virtual Glasses Try-On

This project implements a **virtual glasses try-on application** using OpenCV with real-time webcam input. It detects the user's eyes using Haar cascades with custom filtering to avoid false detections (e.g., mouth or eyebrows), and overlays transparent glasses images onto the face aligned properly with the detected eyes.

---

## Features

- Real-time webcam video feed
- Face and eye detection using OpenCV Haar cascades
- Improved eye filtering to avoid false positives (mouth, eyebrows)
- Multiple glasses support (switch between glasses images)
- Transparent PNG glasses overlay with scaling and positioning
- Smoothed eye position to reduce jitter/shaking
- Keyboard controls:
  - Press **`n`** to switch glasses
  - Press **`q`** to quit the application

---

## Requirements

- Python 3.x
- OpenCV for Python

Install OpenCV via pip:

```bash
pip install opencv-python
