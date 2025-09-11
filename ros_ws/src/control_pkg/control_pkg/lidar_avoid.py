#!/usr/bin/env python3
import math
import numpy as np

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from std_msgs.msg import String


class LidarAvoid(Node):
    def __init__(self):
        super().__init__('lidar_avoid')

        # Parámetros
        self.declare_parameter('scan_topic', '/scan')
        self.declare_parameter('cmd_topic', '/cmd/lidar')
        self.declare_parameter('stop_dist', 0.30)       # m: freno directo si hay algo muy cerca al frente
        self.declare_parameter('clear_dist', 0.60)      # m: si el máximo es menor a esto, consideramos "todo cerca"
        self.declare_parameter('deadband_deg', 15.0)    # ±deg alrededor del frente para ir recto
        self.declare_parameter('avg_window', 7)         # suavizado (ventana impar)
        self.declare_parameter('front_halfwidth_deg', 20.0)  # sector frontal para stop
        self.declare_parameter('min_valid', 0.05)       # m
        self.declare_parameter('max_clip', 6.0)         # m
        self.declare_parameter('back_up_if_all_close', True)

        scan_topic  = self.get_parameter('scan_topic').get_parameter_value().string_value
        cmd_topic   = self.get_parameter('cmd_topic').get_parameter_value().string_value
        self.stop_d = float(self.get_parameter('stop_dist').value)
        self.clear_d= float(self.get_parameter('clear_dist').value)
        self.deadband = math.radians(float(self.get_parameter('deadband_deg').value))
        self.win = int(self.get_parameter('avg_window').value)
        if self.win % 2 == 0:
            self.win += 1
        self.front_hw = math.radians(float(self.get_parameter('front_halfwidth_deg').value))
        self.min_valid = float(self.get_parameter('min_valid').value)
        self.max_clip  = float(self.get_parameter('max_clip').value)
        self.back_up   = bool(self.get_parameter('back_up_if_all_close').value)

        self.pub = self.create_publisher(String, cmd_topic, 10)
        self.sub = self.create_subscription(LaserScan, scan_topic, self.on_scan, 10)

        self.last_cmd = None
        self.get_logger().info(f"LidarAvoid: sub {scan_topic} → pub {cmd_topic}")

    @staticmethod
    def moving_average_circular(x: np.ndarray, w: int) -> np.ndarray:
        if w <= 1:
            return x
        k = np.ones(w, dtype=np.float32) / w
        y = np.convolve(np.concatenate([x, x, x]), k, mode='same')
        return y[len(x):2*len(x)]

    def publish_cmd(self, c: str):
        if c != self.last_cmd:
            self.pub.publish(String(data=c))
            if c != 'S':
                self.get_logger().info(f"/cmd/lidar → {c}")
            self.last_cmd = c

    def on_scan(self, msg: LaserScan):
        r = np.array(msg.ranges, dtype=np.float32)
        r[np.isnan(r)] = self.max_clip
        r[np.isinf(r)] = self.max_clip
        r = np.clip(r, self.min_valid, self.max_clip)

        # STOP si algo muy cerca en el frente ±front_hw
        hw_idx = int(self.front_hw / msg.angle_increment)
        # índice del frente (ángulo 0)
        center_idx = int((-msg.angle_min) / msg.angle_increment) % len(r)
        left = (center_idx - hw_idx) % len(r)
        right = (center_idx + hw_idx) % len(r)
        if left <= right:
            front_min = float(np.min(r[left:right+1]))
        else:
            front_min = float(min(np.min(r[:right+1]), np.min(r[left:])))

        if front_min < self.stop_d:
            self.publish_cmd('S')
            return

        # Suavizado y búsqueda del ángulo con mayor distancia
        rs = self.moving_average_circular(r, self.win)
        best_idx = int(np.argmax(rs))
        best_angle = msg.angle_min + best_idx * msg.angle_increment  # rad
        # normaliza a [-pi, pi]
        a = math.atan2(math.sin(best_angle), math.cos(best_angle))

        # Si todo está "cerca", opcionalmente retrocede
        if self.back_up and float(np.max(rs)) < self.clear_d:
            self.publish_cmd('B')
            return

        # Decide dirección: cercano al frente → F, si no gira hacia el signo del ángulo
        if abs(a) <= self.deadband:
            self.publish_cmd('F')
        else:
            self.publish_cmd('L' if a > 0.0 else 'R')


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
