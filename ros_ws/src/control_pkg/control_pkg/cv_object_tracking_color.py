import os
import sys
import time
from collections import deque
from datetime import datetime
from typing import List, Tuple

import cv2
import numpy as np

import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Bool, Float32
from std_msgs.msg import Float32MultiArray

VALID = {'F', 'B', 'L', 'R', 'S'}
try:
    import RPi.GPIO as GPIO
except Exception:
    GPIO = None


def remove_local_ncnn_from_path():
    """Avoid picking the source tree at ~/ncnn/python instead of the pip wheel."""
    sys.path = [p for p in sys.path if not p.endswith("/ncnn/python")]


class VisionTracker(Node):
    """
    Rastreador por detección (YOLO NCNN) que publica comandos a /cmd/vision:
      - 'S' si no hay objetivo
      - 'F' si está centrado (avanzar)
      - 'L' si el objetivo está a la izquierda
      - 'R' si el objetivo está a la derecha
    """

    def __init__(self):
        super().__init__('vision_tracker')

        # Tópicos y cámara
        self.declare_parameter('vision_topic', '/cmd/vision')
        self.declare_parameter('camera_device_id', 0)
        self.declare_parameter('image_width', 640)
        self.declare_parameter('image_height', 480)
        self.declare_parameter('camera_fps', 30)

        # Modelo YOLO (NCNN)
        self.declare_parameter('yolo_param', '/home/mario/OpenCV_VC/ros_ws/src/yolo_pkg/models/best.param')
        self.declare_parameter('yolo_bin', '/home/mario/OpenCV_VC/ros_ws/src/yolo_pkg/models/best.bin')
        self.declare_parameter('model_input_size', 480)
        self.declare_parameter('conf_thresh', 0.60)
        self.declare_parameter('nms_thresh', 0.45)
        self.declare_parameter('class_names', ['obj0', 'obj1', 'obj3'])
        self.declare_parameter('yolo_threads', 4)

        # Lógica de seguimiento
        self.declare_parameter('trail_len', 32)
        self.declare_parameter('smooth_alpha', 0.25)
        self.declare_parameter('center_tol_px', 60)
        self.declare_parameter('publish_hz', 20.0)
        self.declare_parameter('keepalive_sec', 0.30)
        self.declare_parameter('search_rotation_sec', 4)
        self.declare_parameter('search_direction', 'L')
        self.declare_parameter('ladder_forward_sec', 2.0)
        self.declare_parameter('ladder_turn_sec', 3)
        self.declare_parameter('ladder_repeats', 3)
        self.declare_parameter('ladder_expand_step_sec', 0.0)
        self.declare_parameter('disable_camera', False)
        self.declare_parameter('use_gpio_start', False)
        self.declare_parameter('gpio_pin', 17)

        # Capturas
        self.declare_parameter('center_frames_for_capture', 10)
        self.declare_parameter('capture_cooldown_sec', 3.0)
        self.declare_parameter('use_drive', True)
        self.declare_parameter('drive_folder_id', '1EdP-E2N8aJFVE3lpVX8mbdzueAb6ceeB')
        self.declare_parameter('credentials_file', '/home/mario/OpenCV_VC/cv_testing/credentials.json')
        self.declare_parameter('token_file', '/home/mario/OpenCV_VC/cv_testing/token.json')
        self.declare_parameter('upload_retries', 3)
        self.declare_parameter('retry_backoff_sec', 1.5)

        # Parámetros -> atributos
        self.vision_topic = self.get_parameter('vision_topic').get_parameter_value().string_value
        self.camera_device_id = self._get_param_any('camera_device_id')
        self.image_width = int(self.get_parameter('image_width').value)
        self.image_height = int(self.get_parameter('image_height').value)
        self.camera_fps = int(self.get_parameter('camera_fps').value)

        self.model_param_path = self.get_parameter('yolo_param').get_parameter_value().string_value
        self.model_bin_path = self.get_parameter('yolo_bin').get_parameter_value().string_value
        self.model_input_size = int(self.get_parameter('model_input_size').value)
        self.conf_thresh = float(self.get_parameter('conf_thresh').value)
        self.nms_thresh = float(self.get_parameter('nms_thresh').value)
        self.yolo_threads = int(self.get_parameter('yolo_threads').value)
        self.class_names = list(self.get_parameter('class_names').value)

        self.trail_len = int(self.get_parameter('trail_len').value)
        self.smooth_alpha = float(self.get_parameter('smooth_alpha').value)
        self.center_tol_px = int(self.get_parameter('center_tol_px').value)
        self.publish_hz = float(self.get_parameter('publish_hz').value)
        self.keepalive_sec = float(self.get_parameter('keepalive_sec').value)
        self.search_rotation_sec = float(self.get_parameter('search_rotation_sec').value)
        self.search_direction = self.get_parameter('search_direction').get_parameter_value().string_value.upper()
        if self.search_direction not in ('L', 'R'):
            self.get_logger().warning(f"search_direction inválido '{self.search_direction}', usando 'L'")
            self.search_direction = 'L'
        self.ladder_forward_sec = float(self.get_parameter('ladder_forward_sec').value)
        self.ladder_turn_sec = float(self.get_parameter('ladder_turn_sec').value)
        self.ladder_repeats = int(self.get_parameter('ladder_repeats').value)
        self.ladder_expand_step_sec = float(self.get_parameter('ladder_expand_step_sec').value)
        self.disable_camera = bool(self.get_parameter('disable_camera').value)
        self.use_gpio_start = bool(self.get_parameter('use_gpio_start').value)
        self.gpio_pin = int(self.get_parameter('gpio_pin').value)
        self.use_start_topic = False  # se activará al recibir datos de /start_gate
        self.start_ready = True
        self.battery_level = -1.0

        self.center_frames_for_capture = int(self.get_parameter('center_frames_for_capture').value)
        self.capture_cooldown_sec = float(self.get_parameter('capture_cooldown_sec').value)
        self.use_drive = bool(self.get_parameter('use_drive').value)
        self.drive_folder_id = self.get_parameter('drive_folder_id').get_parameter_value().string_value
        self.credentials_file = self.get_parameter('credentials_file').get_parameter_value().string_value
        self.token_file = self.get_parameter('token_file').get_parameter_value().string_value
        self.upload_retries = int(self.get_parameter('upload_retries').value)
        self.retry_backoff_sec = float(self.get_parameter('retry_backoff_sec').value)

        # Estado
        self.track_points = deque(maxlen=self.trail_len)
        self.smoothed_center = None
        self.last_cmd = None
        self.last_pub_t = 0.0
        self.centered_counter = 0
        self.last_capture_t = 0.0
        self.fps = 0.0
        self.num_classes = None
        self.search_started_at = None  # se inicia en el primer tick
        self.search_completed = False
        self.ladder_plan = deque()
        self.ladder_active = False
        self.ladder_cmd = None
        self.ladder_cmd_until = 0.0
        self.ladder_iteration = 0
        self.last_target_log_t = 0.0
        self.target_log_interval = 0.8
        self.gpio_ready = not self.use_gpio_start
        self.gpio_value_fd = None
        self.gpio_mode = "none"  # "rpi" o "sysfs"
        if self.use_gpio_start:
            self._init_gpio_gate()

        # Inicialización
        self.IS_RASPI_CAMERA = self._is_raspberry_camera()
        self.drive = None
        if self.use_drive:
            self._init_drive()

        if not self.disable_camera:
            self._load_yolo()
            self.cap, backend = self.open_capture()
        else:
            self.cap = None
            backend = "DISABLED"

        self.pub_cmd = self.create_publisher(String, self.vision_topic, 10)
        self.sub_start = self.create_subscription(Bool, '/start_gate', self._cb_start_gate, 10)
        self.sub_batt = self.create_subscription(Float32, '/battery', self._cb_batt, 10)
        self.pub_stats = self.create_publisher(Float32MultiArray, '/vision/stats', 10)
        self.get_logger().info(f"[Vision] Backend: {backend}")

        cv2.namedWindow('frame', cv2.WINDOW_AUTOSIZE)

        period = 1.0 / max(1.0, self.publish_hz)
        self.timer = self.create_timer(period, self.tick)

    # ---------------------- Inicialización ---------------------- #
    def _load_yolo(self):
        if self.disable_camera:
            return
        remove_local_ncnn_from_path()
        try:
            import ncnn
        except Exception as e:
            raise RuntimeError(f"No se pudo importar ncnn: {e}")

        if not os.path.exists(self.model_param_path) or not os.path.exists(self.model_bin_path):
            raise FileNotFoundError(f"Modelos YOLO no encontrados: {self.model_param_path}, {self.model_bin_path}")

        self.ncnn = ncnn
        self.net = ncnn.Net()
        self.net.opt.use_vulkan_compute = False
        self.net.opt.num_threads = max(1, self.yolo_threads)
        self.net.opt.use_packing_layout = True
        self.net.opt.use_sgemm_convolution = True
        self.net.opt.use_winograd_convolution = True
        self.net.opt.use_fp16_storage = True
        self.net.opt.use_fp16_arithmetic = False

        if self.net.load_param(self.model_param_path):
            raise RuntimeError(f"Error cargando param {self.model_param_path}")
        if self.net.load_model(self.model_bin_path):
            raise RuntimeError(f"Error cargando bin {self.model_bin_path}")

        self.get_logger().info(f"[YOLO] Modelo cargado ({self.model_param_path})")

    def _init_drive(self):
        try:
            from pydrive2.auth import GoogleAuth
            from pydrive2.drive import GoogleDrive
        except Exception as e:
            self.get_logger().warning(f"PyDrive2 no disponible: {e}. Subida deshabilitada.")
            self.use_drive = False
            return

        try:
            if not os.path.exists(self.credentials_file):
                raise FileNotFoundError(f"Credenciales no encontradas: {self.credentials_file}")

            gauth = GoogleAuth()
            gauth.LoadClientConfigFile(self.credentials_file)
            if os.path.exists(self.token_file):
                gauth.LoadCredentialsFile(self.token_file)
            if (not gauth.credentials) or gauth.access_token_expired:
                gauth.LocalWebserverAuth()
                gauth.SaveCredentialsFile(self.token_file)

            self.drive = GoogleDrive(gauth)
            folder = self.drive.CreateFile({'id': self.drive_folder_id})
            folder.FetchMetadata(fields='id,title,mimeType')
            if folder['mimeType'] != 'application/vnd.google-apps.folder':
                raise ValueError("drive_folder_id no corresponde a una carpeta.")
            self.get_logger().info("Drive listo: carpeta verificada.")
        except Exception as e:
            self.get_logger().warning(f"No se pudo inicializar Drive: {e}. Subida deshabilitada.")
            self.use_drive = False
            self.drive = None

    def _upload_to_drive(self, local_path, meta_desc=""):
        if not self.use_drive or self.drive is None:
            return False
        try:
            filename = os.path.basename(local_path)
            last_err = None
            for i in range(1, self.upload_retries + 1):
                try:
                    f = self.drive.CreateFile({
                        'title': filename,
                        'parents': [{'id': self.drive_folder_id}],
                        'description': meta_desc
                    })
                    f.SetContentFile(local_path)
                    f.Upload()
                    self.get_logger().info(f"[Drive] Subida exitosa: {filename}")
                    return True
                except Exception as e:
                    last_err = e
                    if i < self.upload_retries:
                        time.sleep(self.retry_backoff_sec * i)
            self.get_logger().warning(f"[Drive] Falló subir {filename}: {last_err}")
            return False
        except Exception as e:
            self.get_logger().warning(f"[Drive] Error inesperado subiendo {local_path}: {e}")
            return False

    def _is_raspberry_camera(self):
        sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        try:
            from utils.picamera_utils import is_raspberry_camera, get_picamera  # noqa
            self._get_picamera = get_picamera
            return is_raspberry_camera()
        except Exception:
            self._get_picamera = None
            return False

    def open_capture(self):
        if self.disable_camera:
            return None, "DISABLED"
        if self.IS_RASPI_CAMERA:
            cam = self._get_picamera(self.image_width, self.image_height)
            cam.start()
            return cam, "RASPI"
        cap = cv2.VideoCapture(self.camera_device_id, cv2.CAP_V4L2)
        if not cap.isOpened():
            cap = cv2.VideoCapture(self.camera_device_id)
        if not cap.isOpened():
            raise IOError(f"Cannot open camera {self.camera_device_id}")
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.image_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.image_height)
        cap.set(cv2.CAP_PROP_FPS, self.camera_fps)
        cap.set(cv2.CAP_PROP_CONVERT_RGB, 1)
        return cap, "V4L2/ANY"

    def read_frame(self):
        if self.disable_camera:
            return np.zeros((self.image_height, self.image_width, 3), dtype=np.uint8)
        if self.IS_RASPI_CAMERA:
            return self.cap.capture_array()
        ret, frame = self.cap.read()
        if not ret:
            return None
        return frame

    def _get_param_any(self, name):
        return self.get_parameter(name).value

    # ---------------------- Detección ---------------------- #
    def letterbox(self, image: np.ndarray, size: int) -> Tuple[np.ndarray, float, Tuple[int, int]]:
        h, w = image.shape[:2]
        scale = min(size / w, size / h)
        new_w, new_h = int(round(w * scale)), int(round(h * scale))
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        pad_w, pad_h = size - new_w, size - new_h
        top, bottom = pad_h // 2, pad_h - pad_h // 2
        left, right = pad_w // 2, pad_w - pad_w // 2
        padded = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))
        return padded, scale, (left, top)

    def run_yolo(self, frame_bgr: np.ndarray):
        if self.disable_camera:
            return []
        img, scale, (pad_x, pad_y) = self.letterbox(frame_bgr, self.model_input_size)
        ncnn = self.ncnn

        # ncnn espera BGR, normalizado 0-1
        mat = ncnn.Mat.from_pixels(img.tobytes(), ncnn.Mat.PixelType.PIXEL_BGR, img.shape[1], img.shape[0])
        mat.substract_mean_normalize([], [1 / 255.0, 1 / 255.0, 1 / 255.0])

        ex = self.net.create_extractor()
        ex.input('in0', mat)
        ret, out = ex.extract('out0')
        if ret != 0:
            self.get_logger().warning(f"YOLO extract falló (ret={ret})")
            return []

        preds = np.array(out)  # (7, 8400) => (num_attrs, num_candidates)
        num_attrs, num_candidates = preds.shape
        if num_attrs < 6:
            return []
        # Formato esperado: [cx, cy, w, h, obj, cls0, cls1, ...]
        boxes_xywh = preds[:4, :].T
        obj = preds[4, :]
        cls_scores = preds[5:, :]
        if self.num_classes is None:
            self.num_classes = cls_scores.shape[0]
            if not self.class_names or len(self.class_names) != self.num_classes:
                self.class_names = [f"cls{i}" for i in range(self.num_classes)]
            self.get_logger().info(f"[YOLO] Clases detectadas: {self.num_classes}")

        cls_best = cls_scores.argmax(axis=0)
        cls_conf = cls_scores.max(axis=0)
        conf = obj * cls_conf

        keep = conf > self.conf_thresh
        if not np.any(keep):
            return []

        boxes_xywh = boxes_xywh[keep]
        conf = conf[keep]
        cls_best = cls_best[keep]

        boxes_xyxy = self.xywh_to_xyxy(boxes_xywh, scale, pad_x, pad_y, frame_bgr.shape[1], frame_bgr.shape[0])
        detections = []
        for box, score, cls_id in zip(boxes_xyxy, conf, cls_best):
            detections.append({
                "bbox": box,  # x1,y1,x2,y2
                "score": float(score),
                "cls": int(cls_id),
                "label": self.class_names[int(cls_id)] if int(cls_id) < len(self.class_names) else f"cls{cls_id}",
            })
        return self.non_max_suppression(detections, self.nms_thresh)

    def xywh_to_xyxy(self, boxes: np.ndarray, scale: float, pad_x: int, pad_y: int, orig_w: int, orig_h: int):
        xyxy = np.zeros_like(boxes)
        xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] * 0.5  # x1
        xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] * 0.5  # y1
        xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] * 0.5  # x2
        xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] * 0.5  # y2

        # Deshacer letterbox
        xyxy[:, [0, 2]] -= pad_x
        xyxy[:, [1, 3]] -= pad_y
        xyxy[:, :4] /= scale

        # Clampeo a frame original
        xyxy[:, 0] = np.clip(xyxy[:, 0], 0, orig_w - 1)
        xyxy[:, 1] = np.clip(xyxy[:, 1], 0, orig_h - 1)
        xyxy[:, 2] = np.clip(xyxy[:, 2], 0, orig_w - 1)
        xyxy[:, 3] = np.clip(xyxy[:, 3], 0, orig_h - 1)
        return xyxy

    def non_max_suppression(self, dets: List[dict], nms_thresh: float) -> List[dict]:
        if not dets:
            return []
        dets = sorted(dets, key=lambda d: d["score"], reverse=True)
        kept = []
        while dets:
            cur = dets.pop(0)
            kept.append(cur)
            dets = [d for d in dets if self.iou(cur["bbox"], d["bbox"]) < nms_thresh]
        return kept

    @staticmethod
    def iou(box1, box2):
        x1, y1, x2, y2 = box1
        xa1, ya1, xa2, ya2 = box2
        inter_x1 = max(x1, xa1)
        inter_y1 = max(y1, ya1)
        inter_x2 = min(x2, xa2)
        inter_y2 = min(y2, ya2)
        inter_w = max(0.0, inter_x2 - inter_x1)
        inter_h = max(0.0, inter_y2 - inter_y1)
        inter_area = inter_w * inter_h
        area1 = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        area2 = max(0.0, xa2 - xa1) * max(0.0, ya2 - ya1)
        union = area1 + area2 - inter_area + 1e-6
        return inter_area / union

    # ---------------------- HUD ---------------------- #
    def put_fps(self, image, fps_val):
        color = (0, 255, 0) if image.ndim == 3 else (255, 255, 255)
        cv2.putText(image, f"FPS={fps_val:.1f}", (8, 18), cv2.FONT_HERSHEY_PLAIN, 1, color, 1, cv2.LINE_AA)

    def put_hud(self, image, status, direction_msg, target_text):
        y = 32
        for line in [
            f"STATUS: {status}",
            f"DIRECTION: {direction_msg}",
            f"TARGET: {target_text}",
            "ESC para salir",
        ]:
            cv2.putText(image, line, (8, y), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1, cv2.LINE_AA)
            y += 18

    # ---------------------- Loop ---------------------- #
    def tick(self):
        start_t = time.time()
        if self.use_start_topic and not self.start_ready:
            ros_now = self.get_clock().now().nanoseconds / 1e9
            if (self.last_cmd != 'S') or ((ros_now - self.last_pub_t) >= self.keepalive_sec):
                self.pub_cmd.publish(String(data='S'))
                self.last_cmd = 'S'
                self.last_pub_t = ros_now
            return

        if self.use_gpio_start and not self.gpio_ready and not self.use_start_topic:
            if self._read_gpio_high():
                self.gpio_ready = True
                self.get_logger().info("GPIO gate en HIGH: iniciando lógica de visión.")
            else:
                ros_now = self.get_clock().now().nanoseconds / 1e9
                if (self.last_cmd != 'S') or ((ros_now - self.last_pub_t) >= self.keepalive_sec):
                    self.pub_cmd.publish(String(data='S'))
                    self.last_cmd = 'S'
                    self.last_pub_t = ros_now
                return

        frame = self.read_frame()
        if frame is None:
            self.get_logger().warning("Failed to read frame")
            return

        detections = self.run_yolo(frame)
        best = detections[0] if detections else None

        vis = frame.copy()
        H, W = vis.shape[:2]
        err = None
        direction_msg = "LOST"
        out_cmd = 'S'
        status = "BUSCANDO"
        centered = False

        now = time.time()
        if self.search_started_at is None:
            self.search_started_at = now
            self.get_logger().info("[Vision] Búsqueda 360 iniciada.")

        searching = False
        if not self.search_completed:
            searching = (now - self.search_started_at) < self.search_rotation_sec
            if not searching:
                self.search_completed = True
                self.get_logger().info("[Vision] Búsqueda 360 completada, entrando en seguimiento normal.")
        if searching:
            out_cmd = self.search_direction
            dir_label = "LEFT" if self.search_direction == 'L' else "RIGHT"
            direction_msg = f"SEARCH ROTATING {dir_label}"
            status = "BUSCANDO 360"

        ladder_override = False
        if self.search_completed and not self.ladder_active and best is None:
            # inicia ladder plan si no hay detección
            self._reset_ladder_plan()
            ladder_override = self._advance_ladder(now)
        elif self.ladder_active:
            ladder_override = self._advance_ladder(now)

        if ladder_override:
            out_cmd = self.ladder_cmd
            direction_msg = f"LADDER {out_cmd}"
            status = "BUSCANDO AMPLIO"

        if best:
            x1, y1, x2, y2 = best["bbox"]
            cx = int((x1 + x2) * 0.5)
            cy = int((y1 + y2) * 0.5)
            target_center = (cx, cy)
            status = "SIGUIENDO"

            if self.smoothed_center is None:
                self.smoothed_center = target_center
            else:
                sx = int((1 - self.smooth_alpha) * self.smoothed_center[0] + self.smooth_alpha * target_center[0])
                sy = int((1 - self.smooth_alpha) * self.smoothed_center[1] + self.smooth_alpha * target_center[1])
                self.smoothed_center = (sx, sy)
            self.track_points.append(self.smoothed_center)

            err = (self.smoothed_center[0] - W // 2, self.smoothed_center[1] - H // 2)
            dx = err[0]
            if searching:
                out_cmd = self.search_direction
                direction_msg = f"SEARCH ROTATING {self.search_direction}"
            else:
                if dx > self.center_tol_px:
                    out_cmd = 'R'; direction_msg = "RIGHT"
                elif dx < -self.center_tol_px:
                    out_cmd = 'L'; direction_msg = "LEFT"
                else:
                    out_cmd = 'F'; direction_msg = "CENTERED→FORWARD"; centered = True

            # Dibujos
            cv2.rectangle(vis, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            label = f"{best['label']} {best['score']:.2f}"
            cv2.putText(vis, label, (int(x1), max(0, int(y1) - 6)), cv2.FONT_HERSHEY_PLAIN, 1.0, (0, 255, 0), 1)
            cv2.circle(vis, self.smoothed_center, 5, (0, 0, 255), -1)
            cv2.line(vis, (W // 2, H // 2), self.smoothed_center, (255, 0, 0), 1)
            self._log_target(best, err, out_cmd, searching)
        else:
            self.smoothed_center = None
            if len(self.track_points) > 0:
                self.track_points.append(self.track_points[-1])
            # Si no hay detección, mantener estado de ladder si está activo

        for i in range(1, len(self.track_points)):
            if self.track_points[i - 1] is None or self.track_points[i] is None:
                continue
            cv2.line(vis, self.track_points[i - 1], self.track_points[i], (0, 200, 255), 2)

        # Captura cuando está centrado
        if centered and status == "SIGUIENDO":
            self.centered_counter += 1
        else:
            self.centered_counter = 0

        if self.centered_counter >= self.center_frames_for_capture:
            if (now - self.last_capture_t) >= self.capture_cooldown_sec:
                ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                path = f"capture_{ts}.jpg"
                cv2.imwrite(path, vis)
                self.last_capture_t = now
                self.centered_counter = 0

                cx_meta = self.smoothed_center[0] if self.smoothed_center else -1
                cy_meta = self.smoothed_center[1] if self.smoothed_center else -1
                meta_desc = (
                    f"timestamp={ts}, cx={cx_meta}, cy={cy_meta}, "
                    f"best_label={best['label'] if best else 'none'}, best_score={best['score'] if best else 0}"
                )
                self.get_logger().info(f"[Capture] Guardada: {path}")
                if self.use_drive and self.drive is not None:
                    self._upload_to_drive(path, meta_desc=meta_desc)

        target_text = f"{best['label']} {best['score']:.2f}" if best else "none"
        self.put_hud(vis, status, direction_msg, target_text)
        dt = time.time() - start_t
        self.fps = 1.0 / dt if dt > 0 else 0.0
        self.put_fps(vis, self.fps)

        cv2.imshow('frame', vis)

        ros_now = self.get_clock().now().nanoseconds / 1e9
        if (out_cmd != self.last_cmd) or ((ros_now - self.last_pub_t) >= self.keepalive_sec):
            self.pub_cmd.publish(String(data=out_cmd))
            if out_cmd != self.last_cmd:
                self.get_logger().info(f"/cmd/vision → {out_cmd}")
            self.last_cmd = out_cmd
            self.last_pub_t = ros_now

        if cv2.waitKey(1) & 0xFF == 27:
            self.get_logger().info("ESC pressed — shutting down vision_tracker...")
            rclpy.shutdown()

        # Publicar métricas en /vision/stats
        stats_msg = Float32MultiArray()
        latency = (1.0 / self.fps) if self.fps > 0 else 0.0
        obj_count = float(min(len(detections), 3))
        stats_msg.data = [float(self.fps), float(latency), float(self.battery_level), obj_count]
        self.pub_stats.publish(stats_msg)

    # ---------------------- Cleanup ---------------------- #
    def destroy_node(self):
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass
        if self.IS_RASPI_CAMERA:
            try:
                self.cap.close()
            except Exception:
                pass
        else:
            try:
                self.cap.release()
            except Exception:
                pass
        if self.use_gpio_start:
            self._cleanup_gpio()
        super().destroy_node()

    def _cb_start_gate(self, msg: Bool):
        self.start_ready = bool(msg.data)
        # activar uso de start_topic en cuanto llegue el primer dato
        self.use_start_topic = True

    def _cb_batt(self, msg: Float32):
        self.battery_level = float(msg.data)

    # ---------------------- GPIO helpers ---------------------- #
    def _init_gpio_gate(self):
        # Intenta RPi.GPIO primero
        if GPIO is not None:
            try:
                GPIO.setmode(GPIO.BCM)
                GPIO.setup(self.gpio_pin, GPIO.IN, pull_up_down=GPIO.PUD_DOWN)
                self.gpio_mode = "rpi"
                self.get_logger().info(f"GPIO gate (RPi.GPIO) en pin {self.gpio_pin}. Esperando nivel alto.")
                return
            except Exception as e:
                self.get_logger().warning(f"No se pudo inicializar RPi.GPIO: {e}. Probando sysfs.")

        # Fallback a sysfs
        try:
            gpio_dir = f"/sys/class/gpio/gpio{self.gpio_pin}"
            if not os.path.exists(gpio_dir):
                with open("/sys/class/gpio/export", "w") as f:
                    f.write(str(self.gpio_pin))
            # direction puede tardar en aparecer
            for _ in range(5):
                try:
                    with open(os.path.join(gpio_dir, "direction"), "w") as f:
                        f.write("in")
                    break
                except Exception:
                    time.sleep(0.1)
            self.gpio_value_fd = open(os.path.join(gpio_dir, "value"), "r")
            self.gpio_mode = "sysfs"
            self.get_logger().info(f"GPIO gate (sysfs) en pin {self.gpio_pin}. Esperando nivel alto.")
        except Exception as e:
            self.get_logger().warning(f"No se pudo inicializar GPIO (sysfs): {e}. Gate deshabilitado.")
            self.gpio_ready = True

    def _read_gpio_high(self):
        if self.gpio_mode == "rpi":
            try:
                return GPIO.input(self.gpio_pin) == GPIO.HIGH
            except Exception as e:
                self.get_logger().warning(f"Lectura GPIO falló: {e}. Deshabilitando gate.")
                self.gpio_ready = True
                return True
        if self.gpio_mode == "sysfs" and self.gpio_value_fd:
            try:
                self.gpio_value_fd.seek(0)
                val = self.gpio_value_fd.read().strip()
                return val == "1"
            except Exception as e:
                self.get_logger().warning(f"Lectura sysfs GPIO falló: {e}. Deshabilitando gate.")
                self.gpio_ready = True
                return True
        return True  # si no hay gate, dejar pasar

    def _cleanup_gpio(self):
        if self.gpio_mode == "rpi" and GPIO is not None:
            try:
                GPIO.cleanup()
            except Exception:
                pass
        if self.gpio_mode == "sysfs" and self.gpio_value_fd:
            try:
                self.gpio_value_fd.close()
            except Exception:
                pass

    def _log_target(self, det, err, cmd, searching):
        """Throttle logging of target center and command."""
        now = time.time()
        if (now - self.last_target_log_t) < self.target_log_interval:
            return
        cx = int((det["bbox"][0] + det["bbox"][2]) * 0.5)
        cy = int((det["bbox"][1] + det["bbox"][3]) * 0.5)
        if err is None:
            ex = ey = 0
        else:
            ex, ey = err
        mode = "BUSQUEDA" if searching else "SEGUIMIENTO"
        self.get_logger().info(
            f"[Vision] {mode}: objetivo en px=({cx},{cy}), err=({ex},{ey}), cmd={cmd}"
        )
        self.last_target_log_t = now

    # ---------------------- Ladder search ---------------------- #
    def _reset_ladder_plan(self):
        """Crea un plan de búsqueda escalonada (escalera)."""
        self.ladder_plan.clear()
        fwd = max(0.1, self.ladder_forward_sec)
        turn = max(0.1, self.ladder_turn_sec)
        repeats = max(1, self.ladder_repeats)
        for i in range(repeats):
            # Patrón: F, L, R=2*L, L (queda orientado al frente original)
            self.ladder_plan.append(('F', fwd))
            self.ladder_plan.append(('L', turn))
            self.ladder_plan.append(('R', 2 * turn))
            self.ladder_plan.append(('L', turn))
            fwd += max(0.0, self.ladder_expand_step_sec)  # opcional expansión
        self.ladder_active = True
        self.ladder_cmd = None
        self.ladder_cmd_until = 0.0
        self.ladder_iteration = 0
        self.get_logger().info(
            f"[Vision] Ladder search iniciada: forward={self.ladder_forward_sec}s, "
            f"turn={self.ladder_turn_sec}s, repeats={repeats}, expand={self.ladder_expand_step_sec}s"
        )

    def _advance_ladder(self, now: float) -> bool:
        """Avanza el plan escalonado según el tiempo; devuelve True si hay comando activo."""
        if not self.ladder_active:
            return False
        # Si se agotó el tiempo del comando actual, tomar el siguiente
        if now >= self.ladder_cmd_until:
            if not self.ladder_plan:
                # plan terminado
                self.ladder_active = False
                self.ladder_cmd = None
                return False
            cmd, duration = self.ladder_plan.popleft()
            self.ladder_cmd = cmd
            self.ladder_cmd_until = now + duration
            self.ladder_iteration += 1
            self.get_logger().info(f"[Vision] Ladder cmd {self.ladder_iteration}: {cmd} for {duration:.2f}s")
        return True


def main():
    rclpy.init()
    node = VisionTracker()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
