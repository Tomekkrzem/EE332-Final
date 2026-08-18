# Offset Measurement Using Computer Vision

Computer vision tool developed for **EE 332: Introduction to Computer Vision** to measure object geometry and offsets between an object's centroid and its internal features.

## Features

* Detects objects and internal features
* Measures diameter, width, and height
* Calculates feature centroid offsets
* Converts pixel measurements to inches or millimeters
* Handles multiple objects and internal features
* Provides rotationally invariant offset measurements
* Supports static images and live camera input

## Approach

The image-processing pipeline uses **CLAHE, Gaussian and bilateral filtering, Canny edge detection, and morphological closing** to extract object boundaries.

OpenCV contours and image moments are then used to determine object geometry and centroids. Contour hierarchy associates internal features with their corresponding objects.

For rotated objects, offsets are transformed from the image coordinate frame into the object's coordinate frame, allowing measurements to remain consistent regardless of object orientation.

## Results

The system successfully measured geometric properties and feature offsets for multiple objects and remained functional when objects were rotated.

The primary limitation was measurement variation in live video caused by lighting, reflective surfaces, camera alignment, and edge-detection noise.

## Requirements

```bash
pip install numpy opencv-python
```

## Usage

```bash
git clone https://github.com/Tomekkrzem/EE332-Final.git
cd EE332-Final
python main.py
```

## Future Improvements

* Camera calibration
* Controlled lighting and camera positioning
* Measurement stabilization using a Kalman filter
* Automatic physical-unit calibration

## Author

**Tomasz Krzeminski**
EE 332 — Introduction to Computer Vision
Northwestern University, Fall 2025
