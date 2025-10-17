#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import numpy as np

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from std_msgs.msg import String


class LidarAvoid(Node):
    def __init__(self):
        super().__init__('lidar_avoid')

        # Parameters
        self.declare_parameter('scan_topic', '/scan')
        self.declare_parameter('cmd_topic', '/cmd/lidar')

        # Distances
        self.declare_parameter('stop_dist', 0.30)       # m: STOP if something too close in front

        # Front sector
        self.declare_parameter('front_halfwidth_deg', 30.0)  # ±angle around front
        self.declare_parameter('front_offset_deg', 180.0)    # correct where 0° is; 180 reverses

        # Preprocess scan
        self.declare_parameter('discard_n_points', 35)
        self.declare_parameter('min_valid', 0.05)       # m
        self.declare_parameter('max_clip', 6.0)         # m

        # Publish frequency
        self.declare_parameter('republish_sec', 0.25)   # keepalive of S while the obstacle persists

        scan_topic  = self.get_parameter('scan_topic').get_parameter_value().string_value
        cmd_topic   = self.get_parameter('cmd_topic').get_parameter_value().string_value

        self.stop_d     = float(self.get_parameter('stop_dist').value)
        self.front_hw   = math.radians(float(self.get_parameter('front_halfwidth_deg').value))
        self.offset_rad = math.radians(float(self.get_parameter('front_offset_deg').value))
        self.discard_n  = int(self.get_parameter('discard_n_points').value)
        self.min_valid  = float(self.get_parameter('min_valid').value)
        self.max_clip   = float(self.get_parameter('max_clip').value)
        self.republish_sec = float(self.get_parameter('republish_sec').value)

        self.pub = self.create_publisher(String, cmd_topic, 10)
        self.sub = self.create_subscription(LaserScan, scan_topic, self.on_scan, 10)

        self.last_cmd = None
        self.last_pub_t = 0.0
        self.get_logger().info(f"LidarAvoid: sub {scan_topic} → pub {cmd_topic}")

    def publish_cmd(self, c: str):
        now = self.get_clock().now().nanoseconds / 1e9
        if c != self.last_cmd or (now - self.last_pub_t) >= self.republish_sec:
            self.pub.publish(String(data=c))
            self.get_logger().info(f"/cmd/lidar → {c}")
            self.last_cmd = c
            self.last_pub_t = now

    def on_scan(self, msg: LaserScan):
        r = np.array(msg.ranges, dtype=np.float32)
        r[np.isnan(r)] = self.max_clip
        r[np.isinf(r)] = self.max_clip
        r = np.clip(r, self.min_valid, self.max_clip)

        n = len(r)
        if n == 0:
            return
        inc = msg.angle_increment
        if inc == 0.0:
            return

        center_idx = int(((0.0 + self.offset_rad) - msg.angle_min) / inc) % n
        hw_idx = int(self.front_hw / abs(inc))

        left = (center_idx - hw_idx) % n
        right = (center_idx + hw_idx) % n

        if left <= right:
            sector = r[left:right+1]
        else:
            sector = np.concatenate([r[left:], r[:right+1]])

        dn = max(0, self.discard_n)
        if 2 * dn < len(sector):
            sector = sector[dn:len(sector)-dn]

        front_min = float(np.min(sector)) if len(sector) > 0 else self.max_clip

        if front_min <= self.stop_d:
            self.publish_cmd('S')  # Stop if obstacle detected
        else:
            self.publish_cmd('F')  # Move forward if no obstacle
        

def main():
    rclpy.init()
    node = LidarAvoid()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()

