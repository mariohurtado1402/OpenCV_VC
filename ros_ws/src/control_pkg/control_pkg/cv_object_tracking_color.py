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
from std_msgs.msg import String

VALID = {'F', 'B', 'L', 'R', 'S'}


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
        self.declare_parameter('image_width', 854)
        self.declare_parameter('image_height', 480)
        self.declare_parameter('camera_fps', 30)

        # Modelo YOLO (NCNN)
        self.declare_parameter('yolo_param', '/home/mario/OpenCV_VC/best.param')
        self.declare_parameter('yolo_bin', '/home/mario/OpenCV_VC/best.bin')
        self.declare_parameter('model_input_size', 640)
        self.declare_parameter('conf_thresh', 0.30)
        self.declare_parameter('nms_thresh', 0.45)
        self.declare_parameter('class_names', ['obj0', 'obj1', 'obj3'])
        self.declare_parameter('yolo_threads', 4)

        # Lógica de seguimiento
        self.declare_parameter('trail_len', 32)
        self.declare_parameter('smooth_alpha', 0.25)
        self.declare_parameter('center_tol_px', 60)
        self.declare_parameter('publish_hz', 20.0)
        self.declare_parameter('keepalive_sec', 0.30)

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

        # Inicialización
        self.IS_RASPI_CAMERA = self._is_raspberry_camera()
        self.drive = None
        if self.use_drive:
            self._init_drive()

        self._load_yolo()

        self.pub_cmd = self.create_publisher(String, self.vision_topic, 10)
        self.cap, backend = self.open_capture()
        self.get_logger().info(f"[Vision] Backend: {backend}")

        cv2.namedWindow('frame', cv2.WINDOW_AUTOSIZE)

        period = 1.0 / max(1.0, self.publish_hz)
        self.timer = self.create_timer(period, self.tick)

    # ---------------------- Inicialización ---------------------- #
    def _load_yolo(self):
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
        else:
            self.smoothed_center = None
            if len(self.track_points) > 0:
                self.track_points.append(self.track_points[-1])

        for i in range(1, len(self.track_points)):
            if self.track_points[i - 1] is None or self.track_points[i] is None:
                continue
            cv2.line(vis, self.track_points[i - 1], self.track_points[i], (0, 200, 255), 2)

        # Captura cuando está centrado
        now = time.time()
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
        super().destroy_node()


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
