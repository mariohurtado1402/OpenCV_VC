#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan

class ClosestObject(Node):
    def __init__(self):
        super().__init__('closest_object_node')

        # Suscribirse al topic del Lidar (cambia '/scan' si tu topic es diferente)
        self.sub = self.create_subscription(
            LaserScan,
            '/scan',
            self.scan_callback,
            10
        )

    def scan_callback(self, msg: LaserScan):
        # Lista de distancias
        ranges = list(msg.ranges)

        # Filtrar valores inválidos (inf o NaN)
        valid_ranges = [r for r in ranges if msg.range_min < r < msg.range_max]

        if not valid_ranges:
            self.get_logger().warn("No se detectaron objetos válidos")
            return

        # Distancia mínima
        min_distance = min(valid_ranges)

        # Índice del objeto más cercano
        idx = ranges.index(min_distance)

        # Ángulo correspondiente
        angle = msg.angle_min + idx * msg.angle_increment

        self.get_logger().info(
            f"Objeto más cercano a {min_distance:.2f} m en ángulo {angle:.2f} rad"
        )


def main(args=None):
    rclpy.init(args=args)
    node = ClosestObject()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
