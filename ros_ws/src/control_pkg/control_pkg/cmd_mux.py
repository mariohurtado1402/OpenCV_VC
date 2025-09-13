#!/usr/bin/env python3
import time
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

import serial

VALID = {'F','B','L','R','S'}

class CmdMux(Node):
    def __init__(self):
        super().__init__('cmd_mux')

        # === Parámetros ===
        self.declare_parameter('vision_topic', '/cmd/vision')
        self.declare_parameter('lidar_topic',  '/cmd/lidar')
        self.declare_parameter('final_topic',  '/cmd/final')
        self.declare_parameter('vision_timeout', 0.6)   # s
        self.declare_parameter('lidar_timeout',  0.6)   # s
        self.declare_parameter('serial_port', '/dev/ttyACM0')
        self.declare_parameter('serial_baud', 115200)
        self.declare_parameter('print_S', True)        # no spamear S en log

        self.vision_topic   = self.get_parameter('vision_topic').get_parameter_value().string_value
        self.lidar_topic    = self.get_parameter('lidar_topic').get_parameter_value().string_value
        self.final_topic    = self.get_parameter('final_topic').get_parameter_value().string_value
        self.vision_timeout = float(self.get_parameter('vision_timeout').value)
        self.lidar_timeout  = float(self.get_parameter('lidar_timeout').value)
        self.serial_port    = self.get_parameter('serial_port').get_parameter_value().string_value
        self.serial_baud    = int(self.get_parameter('serial_baud').value)
        self.print_S        = bool(self.get_parameter('print_S').value)

        # Estado de entradas
        self.last_vision = ('S', 0.0)   # (cmd, tstamp)
        self.last_lidar  = ('S', 0.0)

        # Serial
        self.ser = None
        try:
            self.ser = serial.Serial(self.serial_port, self.serial_baud, timeout=0.3)
            time.sleep(2.0)
            self.get_logger().info(f"Serial OK {self.serial_port} @ {self.serial_baud}")
        except Exception as e:
            self.get_logger().warn(f"Sin serial: {e}")

        # Pub resultado
        self.pub_final = self.create_publisher(String, self.final_topic, 10)

        # Subs
        self.sub_vision = self.create_subscription(String, self.vision_topic, self.cb_vision, 10)
        self.sub_lidar  = self.create_subscription(String, self.lidar_topic,  self.cb_lidar,  10)

        # Loop de decisión
        self.timer = self.create_timer(0.05, self.tick)  # 20 Hz
        self.last_sent = None

        self.get_logger().info(f"cmd_mux escuchando {self.vision_topic} y {self.lidar_topic}")

    def cb_vision(self, msg: String):
        c = msg.data.strip().upper()
        if c in VALID:
            self.last_vision = (c, time.time())

    def cb_lidar(self, msg: String):
        c = msg.data.strip().upper()
        if c in VALID:
            self.last_lidar = (c, time.time())

    def fresh(self, last_tuple, timeout):
        cmd, t = last_tuple
        return (time.time() - t) <= timeout

    def decide(self):
        v_cmd, v_t = self.last_vision
        l_cmd, l_t = self.last_lidar

        v_fresh = self.fresh(self.last_vision, self.vision_timeout)
        l_fresh = self.fresh(self.last_lidar,  self.lidar_timeout)

        # 1) si cualquiera pide STOP (y está fresco) → S
        if (v_fresh and v_cmd == 'S') or (l_fresh and l_cmd == 'S'):
            return 'S'

        # 2) si LIDAR es fresco y NO dice F (o sea, evasión) → obedece LIDAR
        if l_fresh and l_cmd in {'L','R','B'}:
            return l_cmd

        # 3) si LIDAR dice F (libre) y VISION es fresco → obedece VISION
        if l_fresh and l_cmd == 'F' and v_fresh:
            return v_cmd

        # 4) si VISION no está fresco → usa LIDAR si está fresco
        if l_fresh:
            return l_cmd

        # 5) si nada fresco → S
        return 'S'

    def send_serial(self, c: str):
        if self.ser is None:
            return
        try:
            self.ser.write((c + "\n").encode('ascii'))
            self.ser.flush()
        except Exception as e:
            self.get_logger().warn(f"Serial error: {e}")

    def tick(self):
        cmd = self.decide()
        if cmd != self.last_sent:
            # publicar /cmd/final
            self.pub_final.publish(String(data=cmd))
            # serial
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

