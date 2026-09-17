#!/usr/bin/env python3
"""
Boilerplate controller for the HE bot.

Fetches a shape from the get_shape service and builds a list of waypoints
to trace it. Fill in the control loop to drive the robot through them.
"""

import argparse
import math

import numpy as np
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from std_msgs.msg import Float64MultiArray
from shape_interface.srv import GetShape

# Robot physical dimensions
WHEEL_RADIUS_M = 0.0255
CHASSIS_RADIUS_M = 0.06412
WHEEL_ANGLES_RAD = np.radians([30.0, 150.0, 270.0])

# Wheel <-> body-velocity mapping
_BODY_TO_WHEEL = np.array([
    [np.cos(WHEEL_ANGLES_RAD[0]) / WHEEL_RADIUS_M, np.sin(WHEEL_ANGLES_RAD[0]) / WHEEL_RADIUS_M, CHASSIS_RADIUS_M / WHEEL_RADIUS_M],
    [np.cos(WHEEL_ANGLES_RAD[1]) / WHEEL_RADIUS_M, np.sin(WHEEL_ANGLES_RAD[1]) / WHEEL_RADIUS_M, CHASSIS_RADIUS_M / WHEEL_RADIUS_M],
    [np.cos(WHEEL_ANGLES_RAD[2]) / WHEEL_RADIUS_M, np.sin(WHEEL_ANGLES_RAD[2]) / WHEEL_RADIUS_M, CHASSIS_RADIUS_M / WHEEL_RADIUS_M],
])
_WHEEL_TO_BODY = np.linalg.inv(_BODY_TO_WHEEL)
_CTRL_LIMIT = 30.0      # rad/s

WAYPOINT_TOLERANCE = 0.04   # metres
CIRCLE_SEGMENTS     = 36
POSITION_KP         = 2.5
YAW_HOLD_KP         = 2.0
CONTROL_PERIOD      = 0.02  # 50 Hz


def body_to_wheels(vx, vy, wz):
    """Body-frame (vx, vy, wz) -> wheel angular velocities [left, right, back]."""
    v_body = np.array([vx, vy, wz])
    wheels = _BODY_TO_WHEEL @ v_body
    if _CTRL_LIMIT > 0:
        wheels = np.clip(wheels, -_CTRL_LIMIT, _CTRL_LIMIT)
    return wheels.tolist()


def yaw_from_quat(w, x, y, z):
    """Convert quaternion (w, x, y, z) to yaw angle in radians."""
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)


def _regular_polygon(cx, cy, n_sides, side_length, start_angle=math.pi / 2):
    """Vertices of a regular polygon centred on (cx, cy), closed back to the
    first vertex so the last waypoint returns the robot to where it started
    drawing."""
    r = side_length / (2 * math.sin(math.pi / n_sides))
    pts = [
        (cx + r * math.cos(start_angle + 2 * math.pi * i / n_sides),
         cy + r * math.sin(start_angle + 2 * math.pi * i / n_sides))
        for i in range(n_sides)
    ]
    return pts + [pts[0]]


def build_waypoints(shape_name, data):
    """World-frame waypoints for `shape_name`, as returned by the get_shape
    service: data[0:2] is the shape's centre (x, y); the remaining entries
    are its size parameters (see shape_service.cpp's shape_map)."""
    cx, cy = data[0], data[1]

    if shape_name == "Circle":
        radius = data[2]
        return [
            (cx + radius * math.cos(2 * math.pi * i / CIRCLE_SEGMENTS),
             cy + radius * math.sin(2 * math.pi * i / CIRCLE_SEGMENTS))
            for i in range(1, CIRCLE_SEGMENTS + 1)
        ]

    if shape_name == "Square":
        return _regular_polygon(cx, cy, 4, data[2], start_angle=math.pi / 4)

    if shape_name == "Triangle":
        return _regular_polygon(cx, cy, 3, data[2])

    if shape_name == "Pentagon":
        return _regular_polygon(cx, cy, 5, data[2])

    if shape_name == "Rectangle":
        w, h = data[2], data[3]
        corners = [
            (cx - w / 2, cy - h / 2),
            (cx + w / 2, cy - h / 2),
            (cx + w / 2, cy + h / 2),
            (cx - w / 2, cy + h / 2),
        ]
        return corners + [corners[0]]

    raise ValueError(f"unknown shape '{shape_name}'")


class ShapeController(Node):
    def __init__(self, speed):
        super().__init__("shape_controller")
        self.speed = speed

        self.pose = None        # (x, y, yaw), latest ground truth
        self.start_pose = None  # (x, y, yaw), recorded on first odom message
        self.wp_index = 0
        self.done = False

        self.cmd_pub = self.create_publisher(Float64MultiArray, "/wheel_commands", 10)
        self.odom_sub = self.create_subscription(Odometry, "/odom", self._odom_cb, 10)

        # Fetch shape from get_shape service
        self.shape_name, self.waypoints = self._request_shape()
        self.get_logger().info(f"Assigned shape: '{self.shape_name}' with {len(self.waypoints)} waypoints.")

        # Control loop timer
        self.timer = self.create_timer(CONTROL_PERIOD, self._control_step)

    def _request_shape(self):
        client = self.create_client(GetShape, "get_shape")
        while not client.wait_for_service(timeout_sec=2.0):
            self.get_logger().info("Waiting for get_shape service...")

        future = client.call_async(GetShape.Request())
        rclpy.spin_until_future_complete(self, future)
        response = future.result()
        if response is None or not response.success:
            raise RuntimeError(
                f"get_shape service call failed: {response and response.message}"
            )

        return response.shape_name, build_waypoints(response.shape_name, list(response.data))

    def _odom_cb(self, msg):
        pos = msg.pose.pose.position
        orient = msg.pose.pose.orientation
        yaw = yaw_from_quat(orient.w, orient.x, orient.y, orient.z)
        self.pose = (pos.x, pos.y, yaw)
        if self.start_pose is None:
            self.start_pose = (pos.x, pos.y, yaw)

    def _publish(self, wheels):
        self.cmd_pub.publish(Float64MultiArray(data=wheels))

    def _control_step(self):
        if self.done or self.pose is None or not self.waypoints:
            return

        curr_x, curr_y, curr_yaw = self.pose
        target_x, target_y = self.waypoints[self.wp_index]

        dx_world = target_x - curr_x
        dy_world = target_y - curr_y
        dist = math.hypot(dx_world, dy_world)

        if dist < WAYPOINT_TOLERANCE:
            self.wp_index += 1
            if self.wp_index >= len(self.waypoints):
                self.done = True
                self._publish([0.0, 0.0, 0.0])
                self.get_logger().info("Shape tracing complete!")
                return
            target_x, target_y = self.waypoints[self.wp_index]
            dx_world = target_x - curr_x
            dy_world = target_y - curr_y
            dist = math.hypot(dx_world, dy_world)

        vx_world = POSITION_KP * dx_world
        vy_world = POSITION_KP * dy_world

        speed_mag = math.hypot(vx_world, vy_world)
        if speed_mag > self.speed:
            scale = self.speed / speed_mag
            vx_world *= scale
            vy_world *= scale

        # Transform world frame velocity vector to body frame by -curr_yaw
        cos_y = math.cos(curr_yaw)
        sin_y = math.sin(curr_yaw)
        vx_body = vx_world * cos_y + vy_world * sin_y
        vy_body = -vx_world * sin_y + vy_world * cos_y

        # Yaw hold: maintain initial orientation
        target_yaw = self.start_pose[2] if self.start_pose is not None else 0.0
        dyaw = math.atan2(math.sin(target_yaw - curr_yaw), math.cos(target_yaw - curr_yaw))
        wz = YAW_HOLD_KP * dyaw

        wheels = body_to_wheels(vx_body, vy_body, wz)
        self._publish(wheels)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--speed", type=float, default=0.25,
                         help="max approach speed, m/s")
    args, ros_args = parser.parse_known_args()

    rclpy.init(args=ros_args)
    node = ShapeController(args.speed)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node._publish([0.0, 0.0, 0.0])
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

