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

        # Serial (según tu pedido)
        self.declare_parameter('serial_port', '/dev/ttyUSB1')
        self.declare_parameter('serial_baud', 9600)

        # Duración del pulso por comando
        self.declare_parameter('pulse_duration_sec', 3.0)

        # Lee parámetros
        self.vision_topic = self.get_parameter('vision_topic').get_parameter_value().string_value
        self.lidar_topic  = self.get_parameter('lidar_topic').get_parameter_value().string_value
        self.final_topic  = self.get_parameter('final_topic').get_parameter_value().string_value

        self.serial_port  = self.get_parameter('serial_port').get_parameter_value().string_value
        self.serial_baud  = int(self.get_parameter('serial_baud').value)
        self.pulse_duration = float(self.get_parameter('pulse_duration_sec').value)

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

        # Estado de ejecución (acción en curso)
        self.active_cmd = None          # 'F','B','L','R' o None
        self.active_until = 0.0
        self.last_sent = None

        # Timer de control
        self.timer = self.create_timer(0.02, self.tick)  # 50 Hz

        # Envía STOP al iniciar
        self.force_stop(source="init")

        self.get_logger().info(
            f"cmd_mux escuchando {self.vision_topic} y {self.lidar_topic}; publicando en {self.final_topic}; "
            f"pulso por comando = {self.pulse_duration}s"
        )

    # === Callbacks ===
    def cb_vision(self, msg: String):
        c = msg.data.strip().upper()
        self.get_logger().info(f"RX vision: '{c}'")
        if c in VALID:
            if c == 'S':
                self.force_stop(source="vision")
            else:
                self.start_action(c, source="vision")
        else:
            self.get_logger().warning(f"Comando vision inválido: {c}")

    def cb_lidar(self, msg: String):
        c = msg.data.strip().upper()
        self.get_logger().info(f"RX lidar: '{c}'")
        if c in VALID:
            if c == 'S':
                # LIDAR STOP tiene prioridad: interrumpe acción y manda S
                self.force_stop(source="lidar")
            else:
                # Por si algún día LIDAR emite L/R/B/F; no es el caso ahora, pero lo soporta.
                self.start_action(c, source="lidar")
        else:
            self.get_logger().warning(f"Comando lidar inválido: {c}")

    # === Control de acciones ===
    def start_action(self, c: str, source: str = ""):
        """Inicia/renueva una acción por pulse_duration (F/B/L/R)."""
        now = time.time()
        self.active_cmd = c
        self.active_until = now + self.pulse_duration
        if self.last_sent != c:
            self._publish_and_send(c, note=f"{source} start ({self.pulse_duration:.1f}s)")
        # Si el comando es igual al que ya estaba, solo se extiende el tiempo

    def force_stop(self, source: str = ""):
        """Fuerza STOP inmediato y limpia acción."""
        self.active_cmd = None
        self.active_until = 0.0
        if self.last_sent != 'S':
            self._publish_and_send('S', note=f"{source} stop")

    def tick(self):
        """Chequea expiración de la acción en curso."""
        if self.active_cmd is None:
            return
        if time.time() >= self.active_until:
            self.force_stop(source="timer")

    # === Salida ===
    def _publish_and_send(self, c: str, note: str = ""):
        # Publica /cmd/final
        self.pub_final.publish(String(data=c))
        # Serial
        try:
            if self.ser is not None:
                self.ser.write((c + "\n").encode('ascii'))
                self.ser.flush()
        except Exception as e:
            self.get_logger().warning(f"Serial error: {e}")
        # Log
        suffix = f" [{note}]" if note else ""
        self.get_logger().info(f"FINAL: {c}{suffix}")
        self.last_sent = c


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
