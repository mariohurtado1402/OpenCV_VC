#!/usr/bin/env python3
import os
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy, QoSProfile, ReliabilityPolicy
from nav_msgs.msg import OccupancyGrid
from std_srvs.srv import Trigger


class MapToPng(Node):
    def __init__(self):
        super().__init__('map_to_png')

        self.declare_parameter('map_topic', '/map')
        self.declare_parameter('output_dir', str(Path.home() / 'maps'))
        self.declare_parameter('basename', 'slam_map')
        self.declare_parameter('occupied_threshold', 0.65)
        self.declare_parameter('free_threshold', 0.25)
        self.declare_parameter('flip_y', True)

        map_topic = self.get_parameter('map_topic').get_parameter_value().string_value
        self.output_dir = Path(self.get_parameter('output_dir').get_parameter_value().string_value).expanduser()
        self.basename = self.get_parameter('basename').get_parameter_value().string_value
        self.occupied_threshold = float(self.get_parameter('occupied_threshold').value)
        self.free_threshold = float(self.get_parameter('free_threshold').value)
        self.flip_y = bool(self.get_parameter('flip_y').value)

        self.output_dir.mkdir(parents=True, exist_ok=True)

        qos = QoSProfile(
            depth=1,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
            reliability=ReliabilityPolicy.RELIABLE
        )
        self.sub = self.create_subscription(OccupancyGrid, map_topic, self.on_map, qos_profile=qos)
        self.srv = self.create_service(Trigger, 'save_map', self.on_save_map)

        self.last_map = None
        self.get_logger().info(
            f"MapToPng escuchando en {map_topic}, guardando en {self.output_dir}"
        )

    def on_map(self, msg: OccupancyGrid):
        self.last_map = msg

    def on_save_map(self, request, response):
        if self.last_map is None:
            response.success = False
            response.message = "No hay mapa recibido todavía"
            return response

        msg = self.last_map
        expected_len = msg.info.width * msg.info.height
        if len(msg.data) != expected_len:
            response.success = False
            response.message = f"Tamaño de mapa inválido: {len(msg.data)} != {expected_len}"
            return response

        grid = np.array(msg.data, dtype=np.int16).reshape((msg.info.height, msg.info.width))
        if self.flip_y:
            # OccupancyGrid arranca en la esquina inferior izquierda, voltear Y para visualizar arriba-norte.
            grid = np.flipud(grid)

        img = np.full_like(grid, 205, dtype=np.uint8)
        occ_thresh = int(self.occupied_threshold * 100)
        free_thresh = int(self.free_threshold * 100)

        img[grid >= occ_thresh] = 0          # Ocupado -> negro
        img[(grid >= 0) & (grid <= free_thresh)] = 254  # Libre -> blanco

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{self.basename}_{ts}.png"
        out_path = self.output_dir / filename

        ok = cv2.imwrite(str(out_path), img)
        if not ok:
            response.success = False
            response.message = f"Error al escribir {out_path}"
            return response

        res = msg.info.resolution
        response.success = True
        response.message = f"Mapa guardado en {out_path} (res {res:.3f} m/px)"
        self.get_logger().info(response.message)
        return response


def main(args=None):
    rclpy.init(args=args)
    node = MapToPng()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
