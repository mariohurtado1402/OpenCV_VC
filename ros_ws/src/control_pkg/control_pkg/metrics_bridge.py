#!/usr/bin/env python3
import json
import urllib.request
import urllib.error

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray


class MetricsBridge(Node):
    def __init__(self):
        super().__init__('metrics_bridge')
        self.declare_parameter('host', '192.168.100.55')
        self.declare_parameter('port', 5000)
        self.declare_parameter('api_path', '/data')
        self.declare_parameter('timeout_sec', 2.0)

        self.host = self.get_parameter('host').get_parameter_value().string_value
        self.port = int(self.get_parameter('port').value)
        self.api_path = self.get_parameter('api_path').get_parameter_value().string_value
        self.timeout = float(self.get_parameter('timeout_sec').value)

        self.url = f"http://{self.host}:{self.port}{self.api_path}"

        self.sub = self.create_subscription(Float32MultiArray, '/vision/stats', self.cb_stats, 10)
        self.get_logger().info(f"MetricsBridge publicando a {self.url}")

    def cb_stats(self, msg: Float32MultiArray):
        values = {
        "fps": msg.data[0],
        "latency": msg.data[1],
        "battery": msg.data[2],
        "count": msg.data[3]
}
        data = json.dumps(values).encode('utf-8')
        req = urllib.request.Request(self.url, data=data, headers={'Content-Type': 'application/json'})
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                if resp.status >= 400:
                    self.get_logger().warn(f"HTTP error {resp.status}")
        except Exception as e:
            self.get_logger().warn(f"HTTP send failed: {e}")


def main(args=None):
    rclpy.init(args=args)
    node = MetricsBridge()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
