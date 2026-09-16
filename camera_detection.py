#!/usr/bin/env python3
# Copyright (c) 2026 e-Yantra, IIT Bombay. All rights reserved.

# Team ID:          1090
# Author List:      eYRC Team
# Filename:         camera_detection.py
# Functions:        centre_of_quad(), find_trapezoids(), main()

import math
import time
from collections import deque
import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from shape_interface.srv import PixelToWorld

STREAM_URL = "http://127.0.0.1:8080/stream"
WINDOW, BINARY_WINDOW = "camera_feed", "trapezoid_borders"
FPS_WINDOW = 30
ARENA_X0, ARENA_Y0, ARENA_X1, ARENA_Y1 = 304, 24, 975, 695
SAND_DISTANCE = 18
HOUGH_THRESHOLD, HOUGH_MIN_LENGTH, HOUGH_MAX_GAP = 40, 35, 25
MIN_TRAPEZOID_AREA, PARALLEL_TOLERANCE_DEG = 1200, 7
REPORT_PERIOD_SEC = 0.5


def centre_of_quad(corners):
    """Find area centroid of quadrilateral using image moments."""
    M = cv2.moments(np.round(corners).astype(np.int32))
    return (M["m10"] / M["m00"], M["m01"] / M["m00"]) if abs(M["m00"]) > 1e-5 else (float(corners[:, 0].mean()), float(corners[:, 1].mean()))


def find_trapezoids(frame):
    """Locate the three trapezoid station funnels in one frame."""
    binary = np.zeros(frame.shape[:2], np.uint8)
    trapezoids = []

    crop = frame[ARENA_Y0:ARENA_Y1, ARENA_X0:ARENA_X1]
    lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)
    floor_color = np.array([np.bincount(lab[:, :, c].ravel()).argmax() for c in range(3)], dtype=np.float32)

    dist = np.linalg.norm(lab.astype(np.float32) - floor_color, axis=2)
    closed = cv2.morphologyEx((dist > SAND_DISTANCE).astype(np.uint8) * 255, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    edges = cv2.Canny(closed, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, HOUGH_THRESHOLD, minLineLength=HOUGH_MIN_LENGTH, maxLineGap=HOUGH_MAX_GAP)

    scratch = np.zeros(crop.shape[:2], dtype=np.uint8)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(scratch, (x1, y1), (x2, y2), 255, 3)
    scratch = cv2.morphologyEx(scratch, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))

    contours, hierarchy = cv2.findContours(scratch, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

    def is_parallel(a1, a2):
        d = abs(a1 - a2) % 180
        return min(d, 180 - d) <= PARALLEL_TOLERANCE_DEG

    if hierarchy is not None:
        for i, cnt in enumerate(contours):
            if cv2.contourArea(cnt) < MIN_TRAPEZOID_AREA:
                continue
            approx = cv2.approxPolyDP(cnt, 0.03 * cv2.arcLength(cnt, True), True)
            if len(approx) == 4 and cv2.isContourConvex(approx):
                pts = approx.reshape(4, 2)
                angles = [math.degrees(math.atan2(pts[(j+1)%4, 1] - pts[j, 1], pts[(j+1)%4, 0] - pts[j, 0])) % 180 for j in range(4)]
                parallel_count = is_parallel(angles[0], angles[2]) + is_parallel(angles[1], angles[3])
                if parallel_count == 1:
                    corners_full = pts.astype(np.float64) + [ARENA_X0, ARENA_Y0]
                    cx, cy = centre_of_quad(corners_full)
                    trapezoids.append((cx, cy, corners_full))
                    cv2.polylines(binary, [np.round(corners_full).astype(np.int32)], True, 255, 2)

    cv2.imwrite("task_1a_binary.png", binary)
    return binary, trapezoids


def main():
    rclpy.init()
    node = Node("camera_feed")
    client = node.create_client(PixelToWorld, "pixel_to_world")

    node.get_logger().info("waiting for the pixel_to_world service ...")
    if not client.wait_for_service(timeout_sec=10.0):
        node.get_logger().error("pixel_to_world service is not up.")
        rclpy.shutdown()
        return

    cap = cv2.VideoCapture(STREAM_URL)
    if not cap.isOpened():
        node.get_logger().error(f"could not open {STREAM_URL}")
        rclpy.shutdown()
        return

    stamps = deque(maxlen=FPS_WINDOW)
    fps = 0.0
    last_report = 0.0

    while rclpy.ok():
        ok, frame = cap.read()
        if not ok:
            break

        stamps.append(time.monotonic())
        if len(stamps) >= 2:
            span = stamps[-1] - stamps[0]
            fps = (len(stamps) - 1) / span if span > 0 else 0.0

        binary, trapezoids = find_trapezoids(frame)

        for cx, cy, corners in trapezoids:
            cv2.polylines(frame, [np.round(corners).astype(np.int32)], True, (0, 0, 255), 2)
            cv2.circle(frame, (int(round(cx)), int(round(cy))), 6, (0, 255, 255), -1)

        cv2.putText(frame, f"{fps:5.1f} FPS", (12, 34), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(frame, f"{len(trapezoids)} trapezoids", (12, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.imshow(WINDOW, frame)
        cv2.imshow(BINARY_WINDOW, binary)

        now = time.monotonic()
        if trapezoids and now - last_report >= REPORT_PERIOD_SEC:
            last_report = now
            print(f"\n{len(trapezoids)} trapezoid(s):")
            for cx, cy, _ in sorted(trapezoids, key=lambda t: (t[1], t[0])):
                req = PixelToWorld.Request(pixel_x=float(cx), pixel_y=float(cy))
                future = client.call_async(req)
                rclpy.spin_until_future_complete(node, future, timeout_sec=0.1)
                res = future.result()
                if res is None:
                    print(f"  pixel ({cx:7.2f}, {cy:7.2f})  ->  timed out")
                elif not res.success:
                    print(f"  pixel ({cx:7.2f}, {cy:7.2f})  ->  {res.message}")
                else:
                    print(f"  pixel ({cx:7.2f}, {cy:7.2f})  ->  world ({res.world_x:6.3f}, {res.world_y:6.3f}) m")

        if (cv2.waitKey(1) & 0xFF) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
