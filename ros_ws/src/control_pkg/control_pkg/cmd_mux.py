#!/usr/bin/env python3
import time
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

import serial

VALID = {'F', 'B', 'L', 'R', 'S'}


class CmdMux(Node):
    def __init__(self):
        super().__init__('cmd_mux')

        # === Parámetros ===
        self.declare_parameter('vision_topic', '/cmd/vision')
        self.declare_parameter('lidar_topic',  '/cmd/lidar')
        self.declare_parameter('final_topic',  '/cmd/final')

        # Timeouts (s): cuánto tiempo consideramos "fresco" el último comando recibido
        self.declare_parameter('vision_timeout', 0.6)
        self.declare_parameter('lidar_timeout',  0.6)

        # Serial
        self.declare_parameter('serial_port', '/dev/ttyACM0')
        self.declare_parameter('serial_baud', 115200)

        # Logging
        self.declare_parameter('print_S', True)  # si True, imprime también los 'S' en el log

        # Lee parámetros
        self.vision_topic   = self.get_parameter('vision_topic').get_parameter_value().string_value
        self.lidar_topic    = self.get_parameter('lidar_topic').get_parameter_value().string_value
        self.final_topic    = self.get_parameter('final_topic').get_parameter_value().string_value
        self.vision_timeout = float(self.get_parameter('vision_timeout').value)
        self.lidar_timeout  = float(self.get_parameter('lidar_timeout').value)
        self.serial_port    = self.get_parameter('serial_port').get_parameter_value().string_value
        self.serial_baud    = int(self.get_parameter('serial_baud').value)
        self.print_S        = bool(self.get_parameter('print_S').value)

        # Estado de entradas: (cmd, timestamp)
        self.last_vision = ('S', 0.0)
        self.last_lidar  = ('S', 0.0)

        # Serial
        self.ser = None
        try:
            self.ser = serial.Serial(self.serial_port, self.serial_baud, timeout=0.3)
            time.sleep(2.0)
            self.get_logger().info(f"Serial OK {self.serial_port} @ {self.serial_baud}")
        except Exception as e:
            self.get_logger().warning(f"Sin serial: {e}")

        # Publisher resultado
        self.pub_final = self.create_publisher(String, self.final_topic, 10)

        # Suscriptores
        self.sub_vision = self.create_subscription(String, self.vision_topic, self.cb_vision, 10)
        self.sub_lidar  = self.create_subscription(String, self.lidar_topic,  self.cb_lidar,  10)

        # Loop de decisión
        self.timer = self.create_timer(0.05, self.tick)  # 20 Hz
        self.last_sent = None

        self.get_logger().info(
            f"cmd_mux escuchando {self.vision_topic} (timeout={self.vision_timeout}s) "
            f"y {self.lidar_topic} (timeout={self.lidar_timeout}s); publicando en {self.final_topic}"
        )

    # === Callbacks ===
    def cb_vision(self, msg: String):
        c = msg.data.strip().upper()
        if c in VALID:
            self.last_vision = (c, time.time())

    def cb_lidar(self, msg: String):
        c = msg.data.strip().upper()
        if c in VALID:
            self.last_lidar = (c, time.time())

    # === Utilidades ===
    def fresh(self, last_tuple, timeout):
        _, t = last_tuple
        return (time.time() - t) <= timeout

    def decide(self):
        v_cmd, _ = self.last_vision
        l_cmd, _ = self.last_lidar

        v_fresh = self.fresh(self.last_vision, self.vision_timeout)
        l_fresh = self.fresh(self.last_lidar,  self.lidar_timeout)

        # 1) Si cualquiera (fresco) pide STOP → S
        if (v_fresh and v_cmd == 'S') or (l_fresh and l_cmd == 'S'):
            return 'S'

        # 2) Si LIDAR es fresco y NO dice F (o sea, evasión) → obedece LIDAR
        if l_fresh and l_cmd in {'L', 'R', 'B'}:
            return l_cmd

        # 3) Si LIDAR dice F (libre) y VISION es fresco → obedece VISION
        if l_fresh and l_cmd == 'F' and v_fresh:
            return v_cmd

        # 4) Si VISION no está fresco → usa LIDAR si está fresco
        if l_fresh:
            return l_cmd

        # 5) Si nada fresco → S
        return 'S'

    def send_serial(self, c: str):
        if self.ser is None:
            return
        try:
            self.ser.write((c + "\n").encode('ascii'))
            self.ser.flush()
        except Exception as e:
            self.get_logger().warning(f"Serial error: {e}")

    # === Timer ===
    def tick(self):
        cmd = self.decide()
        if cmd != self.last_sent:
            # Publica /cmd/final
            self.pub_final.publish(String(data=cmd))
            # Envía por serial (si hay)
            self.send_serial(cmd)
            if cmd != 'S' or self.print_S:
                self.get_logger().info(f"FINAL: {cmd}")
            self.last_sent = cmd


def main():
    rclpy.init()
    node = CmdMux()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
