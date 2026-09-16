# eYRC 2026-27: Hola The Explorer - Task 1A (Team ID: 1090)

## Overview
This repository contains the complete implementation for **Task 1A: See the Checkpoint** in the eYRC 2026-27 (Theme: Hola The Explorer) competition.

The objective is to detect three station funnels (trapezoids: cyan, green, orange) from an overhead camera stream, reduce the camera frame to a **1280x720 binary image** holding only the trapezoid outlines, and convert pixel coordinates to world coordinates via the `/pixel_to_world` ROS 2 service.

---

## Technical Features
- **LAB Color Thresholding**: Mode-based distance masking to separate sand floor from features.
- **Edge & Hough Line Extraction**: Uses Canny edge detection and Probabilistic Hough Transform (`cv2.HoughLinesP`).
- **Hierarchy & Geometry Filtering**: `cv2.RETR_CCOMP` inner contour extraction; filters for convex quadrilaterals with exactly 1 pair of parallel opposite sides (rejects rectangular safes and central hexagon).
- **Centroid Calculation**: Uses image area moments (`m10/m00`, `m01/m00`).
- **ROS 2 Integration**: Non-blocking asynchronous queries to `/pixel_to_world` service.

---

## Submission Verification
- **Output Mask Dimensions**: Single channel `(720, 1280)`, `uint8`
- **Binary Values**: `np.unique` strictly `[0, 255]`
- **Contour Count**: Exactly 3 closed outlines

---

## Repository Files
- `HE_1090_camera_detection.py`: Final ROS 2 Python node implementation.
- `HE_1090_binary.png`: Generated 1280x720 binary output mask.
- `HE_1090_task_1a.zip`: Submission zip containing `HE_1090_binary.png` & `HE_1090_camera_detection.py`.
- `create_submission.sh`: Helper script to package submission zip files.
