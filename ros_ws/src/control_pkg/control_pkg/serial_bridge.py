#!/usr/bin/env python3
import time
import serial

import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool, Float32


class SerialBridge(Node):
    def __init__(self):
        super().__init__('serial_bridge')

        self.declare_parameter('serial_port', '/dev/arduino')
        self.declare_parameter('serial_baud', 9600)
        self.declare_parameter('start_topic', '/start_gate')
        self.declare_parameter('battery_topic', '/battery')
        self.declare_parameter('poll_hz', 10.0)

        self.port = self.get_parameter('serial_port').get_parameter_value().string_value
        self.baud = int(self.get_parameter('serial_baud').value)
        self.start_topic = self.get_parameter('start_topic').get_parameter_value().string_value
        self.batt_topic = self.get_parameter('battery_topic').get_parameter_value().string_value
        self.poll_hz = float(self.get_parameter('poll_hz').value)

        self.pub_start = self.create_publisher(Bool, self.start_topic, 10)
        self.pub_batt = self.create_publisher(Float32, self.batt_topic, 10)

        self.ser = None
        self._open_serial()

        period = 1.0 / max(1.0, self.poll_hz)
        self.timer = self.create_timer(period, self.tick)

    def _open_serial(self):
        try:
            self.ser = serial.Serial(self.port, self.baud, timeout=0.1)
            self.get_logger().info(f"Serial abierto: {self.port} @ {self.baud}")
        except Exception as e:
            self.get_logger().warning(f"No se pudo abrir serial {self.port}: {e}")
            self.ser = None

    def tick(self):
        if self.ser is None or not self.ser.is_open:
            self._open_serial()
            return
        try:
            while self.ser.in_waiting:
                line = self.ser.readline().decode('ascii', errors='ignore').strip()
                if not line:
                    continue
                self._parse_line(line)
        except Exception as e:
            self.get_logger().warning(f"Error leyendo serial: {e}")
            try:
                self.ser.close()
            except Exception:
                pass
            self.ser = None

    def _parse_line(self, line: str):
        # Espera formato: START=<0|1>,BATT=<float>
        parts = line.split(',')
        start_val = None
        batt_val = None
        for part in parts:
            if '=' not in part:
                continue
            k, v = part.split('=', 1)
            k = k.strip().upper()
            v = v.strip()
            if k == 'START':
                start_val = v in ('1', 'true', 'TRUE', 'HIGH')
            elif k == 'BATT':
                try:
                    batt_val = float(v)
                except Exception:
                    pass
        if start_val is not None:
            self.pub_start.publish(Bool(data=start_val))
        if batt_val is not None:
            self.pub_batt.publish(Float32(data=batt_val))


def main(args=None):
    rclpy.init(args=args)
    node = SerialBridge()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
