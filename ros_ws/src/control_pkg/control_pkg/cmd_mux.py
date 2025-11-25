import time
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import serial

VALID = {'F', 'B', 'L', 'R', 'S'}

class CmdMux(Node):
    def __init__(self):
        super().__init__('cmd_mux')

        self.declare_parameter('vision_topic', '/cmd/vision')
        self.declare_parameter('lidar_topic',  '/cmd/lidar')
        self.declare_parameter('final_topic',  '/cmd/final')

        self.declare_parameter('serial_port', '/dev/arduino_nano')
        self.declare_parameter('serial_baud', 9600)
        self.declare_parameter('pulse_duration_sec', 3.0)

        self.vision_topic = self.get_parameter('vision_topic').get_parameter_value().string_value
        self.lidar_topic  = self.get_parameter('lidar_topic').get_parameter_value().string_value
        self.final_topic  = self.get_parameter('final_topic').get_parameter_value().string_value
        self.serial_port  = self.get_parameter('serial_port').get_parameter_value().string_value
        self.serial_baud  = int(self.get_parameter('serial_baud').value)
        self.pulse_duration = float(self.get_parameter('pulse_duration_sec').value)

        self.ser = None
        try:
            self.ser = serial.Serial(self.serial_port, self.serial_baud, timeout=0.3)
            time.sleep(2.0)
            self.get_logger().info(f"Serial OK {self.serial_port} @ {self.serial_baud}")
        except Exception as e:
            self.get_logger().warning(f"Serial error: {e}")

        self.pub_final = self.create_publisher(String, self.final_topic, 10)

        self.sub_vision = self.create_subscription(String, self.vision_topic, self.cb_vision, 10)
        self.sub_lidar  = self.create_subscription(String, self.lidar_topic,  self.cb_lidar,  10)

        self.active_cmd = None
        self.active_until = 0.0
        self.last_sent = None
        self.lidar_stop_active = False

        self.timer = self.create_timer(0.02, self.tick)

        self.force_stop(source="init")

        self.get_logger().info(
            f"cmd_mux listening to {self.vision_topic} and {self.lidar_topic}; "
            f"publishing to {self.final_topic}; pulse={self.pulse_duration}s"
        )

    def cb_vision(self, msg: String):
        """Handle vision commands."""
        if self.lidar_stop_active:
            self.get_logger().info("LIDAR is in STOP state, ignoring vision command.")
            return

        c = msg.data.strip().upper()
        self.get_logger().info(f"RX vision: '{c}'")
        if c in VALID:
            if c == 'S':
                self.force_stop(source="vision")
            else:
                self.start_action(c, source="vision")
        else:
            self.get_logger().warning(f"Invalid vision command: {c}")

    def cb_lidar(self, msg: String):
        """Handle LiDAR commands."""
        c = msg.data.strip().upper()
        if c in VALID:
            if c == 'S':
                self.force_stop(source="lidar")
            elif c == 'F':
                self.lidar_stop_active = False
                self.start_action('F', source="lidar")
            else:
                self.start_action(c, source="lidar")
        else:
            self.get_logger().warning(f"Invalid lidar command: {c}")

    def start_action(self, c: str, source: str = ""):
        """Starts/renew action with pulse duration (F/B/L/R)."""
        now = time.time()
        self.active_cmd = c
        self.active_until = now + self.pulse_duration
        if self.last_sent != c:
            self._publish_and_send(c, note=f"{source} start ({self.pulse_duration:.1f}s)")

    def force_stop(self, source: str = ""):
        """Force STOP and clear current action."""
        self.active_cmd = None
        self.active_until = 0.0
        if self.last_sent != 'S':
            self._publish_and_send('S', note=f"{source} stop")
            if source == "lidar":
                self.lidar_stop_active = True

    def tick(self):
        """Check if the current action has expired."""
        if self.active_cmd is None:
            return
        if time.time() >= self.active_until:
            self.force_stop(source="timer")

    def _publish_and_send(self, c: str, note: str = ""):
        self.pub_final.publish(String(data=c))

        try:
            if self.ser is not None:
                self.ser.write((c + "\n").encode('ascii'))
                self.ser.flush()
        except Exception as e:
            self.get_logger().warning(f"Serial error: {e}")

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
