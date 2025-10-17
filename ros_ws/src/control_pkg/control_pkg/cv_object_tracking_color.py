#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import time
from collections import deque
from datetime import datetime

import cv2
import numpy as np

import rclpy
from rclpy.node import Node
from std_msgs.msg import String

# -------------------------------------------------------
# Add src directory to the path (tu comportamiento original)
# -------------------------------------------------------
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    from utils.picamera_utils import is_raspberry_camera, get_picamera
except Exception:
    # Fallback si no están las utils de la Raspi
    def is_raspberry_camera():
        return False
    def get_picamera(w, h):
        raise RuntimeError("Raspi camera utils not available")

# ----------------------------
# Helpers / defaults
# ----------------------------
VALID = {'F', 'B', 'L', 'R', 'S'}

def rgb2hsv_compat(r, g, b):
    """RGB [0..255] -> OpenCV HSV (H:[0..179], S:[0..255], V:[0..255])"""
    r, g, b = r/255.0, g/255.0, b/255.0
    mx = max(r, g, b); mn = min(r, g, b); df = mx - mn
    if df == 0:
        h = 0
    elif mx == r:
        h = (60 * ((g-b)/df) + 360) % 360
    elif mx == g:
        h = (60 * ((b-r)/df) + 120) % 360
    else:
        h = (60 * ((r-g)/df) + 240) % 360
    s = 0 if mx == 0 else df/mx
    v = mx
    return (int(h/2), int(s*255), int(v*255))

def auto_range_from_samples(samples, h_tol, s_tol, v_tol):
    """Crea HSV min/max desde muestras con tolerancias, acotado a rangos válidos."""
    h_vals = [h for h, s, v in samples]
    s_vals = [s for h, s, v in samples]
    v_vals = [v for h, s, v in samples]
    minh = max(0, min(h_vals) - h_tol)
    maxh = min(179, max(h_vals) + h_tol)
    mins = max(0, min(s_vals) - s_tol)
    maxs = min(255, max(s_vals) + s_tol)
    minv = max(0, min(v_vals) - v_tol)
    maxv = min(255, max(v_vals) + v_tol)
    return np.array((minh, mins, minv), dtype=np.uint8), np.array((maxh, maxs, maxv), dtype=np.uint8)


class VisionTracker(Node):
    """
    Rastreador de color con OpenCV que publica comandos a /cmd/vision:
      - 'S' si no hay objetivo
      - 'F' si está centrado (avanzar)
      - 'L' si el objetivo está a la izquierda
      - 'R' si el objetivo está a la derecha

    Incluye:
      - Preset HSV para pelota de tenis + auto-calibración opcional al arrancar
      - HUD con estado BUSCANDO/SIGUIENDO, FPS, y dirección
      - Captura local cuando el objetivo permanece centrado N frames
      - Subida opcional de la captura a Google Drive (PyDrive2)
    """

    def __init__(self):
        super().__init__('vision_tracker')

        # ===== Parámetros =====
        self.declare_parameter('vision_topic', '/cmd/vision')

        # Cámara
        self.declare_parameter('camera_device_id', 0)          # o '/dev/video0'
        self.declare_parameter('image_width', 854)
        self.declare_parameter('image_height', 480)
        self.declare_parameter('camera_fps', 30)

        # Preset / Auto-calibración para pelota de tenis
        self.declare_parameter('tennis_preset_enable', True)
        self.declare_parameter('tennis_hsv_min', [20, 80, 80])     # amarillo-verdoso
        self.declare_parameter('tennis_hsv_max', [45, 255, 255])
        self.declare_parameter('h_tol', 12)
        self.declare_parameter('s_tol', 60)
        self.declare_parameter('v_tol', 60)

        # Morfología
        self.declare_parameter('open_kernel', 3)    # 3x3
        self.declare_parameter('close_kernel', 5)   # 5x5

        # Tracking & UI
        self.declare_parameter('min_blob_area', 150)
        self.declare_parameter('trail_len', 32)
        self.declare_parameter('smooth_alpha', 0.25)   # 0..1
        self.declare_parameter('center_tol_px', 60)    # tolerancia dx para "centrado → F"

        # Publicación / Keepalive
        self.declare_parameter('publish_hz', 20.0)
        self.declare_parameter('keepalive_sec', 0.30)

        # Captura local
        self.declare_parameter('center_frames_for_capture', 10)  # N frames centrado
        self.declare_parameter('capture_cooldown_sec', 3.0)      # enfriamiento entre capturas

        # Google Drive (opcional)
        self.declare_parameter('use_drive', True)
        self.declare_parameter('drive_folder_id', '1EdP-E2N8aJFVE3lpVX8mbdzueAb6ceeB')
        # ajusta estas rutas:
        self.declare_parameter('credentials_file', '/home/mario/OpenCV_VC/cv_testing/credentials.json')
        self.declare_parameter('token_file', '/home/mario/OpenCV_VC/cv_testing/token.json')
        self.declare_parameter('upload_retries', 3)
        self.declare_parameter('retry_backoff_sec', 1.5)

        # ---- Leer parámetros
        self.vision_topic = self.get_parameter('vision_topic').get_parameter_value().string_value
        self.camera_device_id = self._get_param_any('camera_device_id')
        self.image_width  = int(self.get_parameter('image_width').value)
        self.image_height = int(self.get_parameter('image_height').value)
        self.camera_fps   = int(self.get_parameter('camera_fps').value)

        self.TENNIS_PRESET_ENABLE = bool(self.get_parameter('tennis_preset_enable').value)
        tennis_min = self.get_parameter('tennis_hsv_min').value
        tennis_max = self.get_parameter('tennis_hsv_max').value
        self.H_TOL = int(self.get_parameter('h_tol').value)
        self.S_TOL = int(self.get_parameter('s_tol').value)
        self.V_TOL = int(self.get_parameter('v_tol').value)

        k_open = int(self.get_parameter('open_kernel').value)
        k_close = int(self.get_parameter('close_kernel').value)
        self.KERNEL_OPEN  = np.ones((max(1, k_open), max(1, k_open)), np.uint8)
        self.KERNEL_CLOSE = np.ones((max(1, k_close), max(1, k_close)), np.uint8)

        self.min_blob_area = int(self.get_parameter('min_blob_area').value)
        self.trail_len = int(self.get_parameter('trail_len').value)
        self.smooth_alpha = float(self.get_parameter('smooth_alpha').value)
        self.center_tol_px = int(self.get_parameter('center_tol_px').value)

        self.publish_hz = float(self.get_parameter('publish_hz').value)
        self.keepalive_sec = float(self.get_parameter('keepalive_sec').value)

        self.center_frames_for_capture = int(self.get_parameter('center_frames_for_capture').value)
        self.capture_cooldown_sec = float(self.get_parameter('capture_cooldown_sec').value)

        # Google Drive cfg
        self.use_drive = bool(self.get_parameter('use_drive').value)
        self.drive_folder_id = self.get_parameter('drive_folder_id').get_parameter_value().string_value
        self.credentials_file = self.get_parameter('credentials_file').get_parameter_value().string_value
        self.token_file = self.get_parameter('token_file').get_parameter_value().string_value
        self.upload_retries = int(self.get_parameter('upload_retries').value)
        self.retry_backoff_sec = float(self.get_parameter('retry_backoff_sec').value)

        # ---- Estado
        self.IS_RASPI_CAMERA = is_raspberry_camera()
        self.track_points = deque(maxlen=self.trail_len)
        self.picked_hsv = []
        self.frame_for_click = None
        self.smoothed_center = None
        self.fps = 0.0
        self.last_cmd = None
        self.last_pub_t = 0.0

        # Rango HSV inicial (preset tenis)
        self.HSV_MIN = np.array(tennis_min, dtype=np.uint8)
        self.HSV_MAX = np.array(tennis_max, dtype=np.uint8)

        # Contadores de captura
        self.centered_counter = 0
        self.last_capture_t = 0.0

        # Drive handle
        self.drive = None
        if self.use_drive:
            self._init_drive()

        # ---- ROS pub
        self.pub_cmd = self.create_publisher(String, self.vision_topic, 10)

        # ---- Cámara
        self.cap, backend = self.open_capture()
        self.get_logger().info(f"[Vision] Backend: {backend}")
        cv2.namedWindow('frame', cv2.WINDOW_AUTOSIZE)
        cv2.namedWindow('mask', cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback('frame', self.on_mouse_click, None)

        # Auto-calibración opcional en el arranque (a partir de 1-3 frames)
        if self.TENNIS_PRESET_ENABLE:
            ok_grab = False
            for _ in range(3):
                frame0 = self.read_frame()
                if frame0 is not None:
                    ok_grab = True
                    if self._autocalibrate_tennis_from_frame(frame0):
                        break
                time.sleep(0.05)
            if not ok_grab:
                self.get_logger().warning("No frame for auto-calib; using tennis preset as-is.")

        # ---- Timer principal
        period = 1.0 / max(1.0, self.publish_hz)
        self.timer = self.create_timer(period, self.tick)

    # ---- Google Drive ----
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
                # Esto abrirá el navegador la primera vez para autorizar
                gauth.LocalWebserverAuth()
                gauth.SaveCredentialsFile(self.token_file)

            self.drive = GoogleDrive(gauth)

            # Verifica carpeta
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

    # ---- Auto-calibración tenis ----
    def _autocalibrate_tennis_from_frame(self, frame_bgr):
        """Busca el tono 'tenis' en un rango amplio y ajusta HSV_MIN/MAX automáticamente."""
        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
        # Rango amplio para “tennis-like”
        broad_min = np.array([18, 70, 70], dtype=np.uint8)
        broad_max = np.array([55, 255, 255], dtype=np.uint8)
        mask = cv2.inRange(hsv, broad_min, broad_max)

        count = int(cv2.countNonZero(mask))
        if count < 500:  # muy pocos píxeles: mantén el preset
            return False

        H = hsv[:, :, 0][mask > 0].astype(np.int32)
        S = hsv[:, :, 1][mask > 0].astype(np.int32)
        V = hsv[:, :, 2][mask > 0].astype(np.int32)

        # Pico del histograma de H
        hist, _ = np.histogram(H, bins=180, range=(0, 180))
        peak_h = int(np.argmax(hist))

        # Medianas parciales de S,V (10–90 percentil) para robustez
        s_sorted = np.sort(S); v_sorted = np.sort(V)
        s_lo = s_sorted[max(0, int(0.10 * len(s_sorted)))]
        v_lo = v_sorted[max(0, int(0.10 * len(v_sorted)))]

        hmin = max(0, peak_h - self.H_TOL)
        hmax = min(179, peak_h + self.H_TOL)
        smin = max(0, int(0.8 * s_lo))
        vmin = max(0, int(0.8 * v_lo))

        self.HSV_MIN = np.array([hmin, smin, vmin], dtype=np.uint8)
        self.HSV_MAX = np.array([hmax, 255, 255], dtype=np.uint8)

        self.get_logger().info(
            f"[AutoCalib] H≈{peak_h} → HSV: {tuple(self.HSV_MIN)}..{tuple(self.HSV_MAX)} (n={count})"
        )
        return True

    # ---- Cámara ----
    def open_capture(self):
        if self.IS_RASPI_CAMERA:
            cam = get_picamera(self.image_width, self.image_height)
            cam.start()
            return cam, "RASPI"
        else:
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
        else:
            ret, frame = self.cap.read()
            if not ret:
                return None
            return frame

    # ---- UI / clicks ----
    def on_mouse_click(self, event, x, y, flags, userdata):
        if event == cv2.EVENT_LBUTTONUP and self.frame_for_click is not None:
            bgr = self.frame_for_click[y, x].tolist()
            rgb = (bgr[2], bgr[1], bgr[0])
            color_hsv = rgb2hsv_compat(*rgb)
            self.picked_hsv.append(color_hsv)
            self.HSV_MIN, self.HSV_MAX = auto_range_from_samples(
                self.picked_hsv, self.H_TOL, self.S_TOL, self.V_TOL
            )
            self.get_logger().info(
                f"[Pick] RGB={rgb} → HSV={color_hsv} | New HSV range: {tuple(self.HSV_MIN)}..{tuple(self.HSV_MAX)}"
            )

    def put_fps(self, image, fps_val):
        color = (0, 255, 0) if image.ndim == 3 else (255, 255, 255)
        cv2.putText(image, f"FPS={fps_val:.1f}", (8, 18),
                    cv2.FONT_HERSHEY_PLAIN, 1, color, 1, cv2.LINE_AA)

    def put_hud(self, image, center, err, direction_msg="", status="BUSCANDO"):
        H, W = image.shape[:2]
        # Centro de imagen
        cv2.drawMarker(image, (W//2, H//2), (255, 255, 255), cv2.MARKER_CROSS, 16, 1)
        # Centro objetivo
        if center is not None:
            cv2.circle(image, center, 5, (0, 0, 255), -1)
            cv2.line(image, (W//2, H//2), center, (255, 0, 0), 1)
        # Texto
        y = 36
        for line in [
            f"STATUS: {status}",
            f"HSV: min={tuple(int(x) for x in self.HSV_MIN)} max={tuple(int(x) for x in self.HSV_MAX)}",
            (f"err=(dx,dy)={err}" if err is not None else "err=(dx,dy)=(None,None)"),
            (f"DIRECTION: {direction_msg}" if direction_msg else "DIRECTION:"),
            "Click en 'frame' para refinar color | ESC para salir",
        ]:
            cv2.putText(image, line, (8, y), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1, cv2.LINE_AA)
            y += 18

    # ---- Bucle principal ----
    def tick(self):
        start_t = time.time()

        frame = self.read_frame()
        if frame is None:
            self.get_logger().warning("Failed to read frame")
            return

        self.frame_for_click = frame
        frame_blur = cv2.GaussianBlur(frame, (5, 5), 0)
        hsv = cv2.cvtColor(frame_blur, cv2.COLOR_BGR2HSV)

        mask = cv2.inRange(hsv, self.HSV_MIN, self.HSV_MAX)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.KERNEL_OPEN, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.KERNEL_CLOSE, iterations=2)

        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        target_center, bbox = None, None
        status = "BUSCANDO"

        if cnts:
            best = max(cnts, key=cv2.contourArea)
            area = cv2.contourArea(best)
            if area > self.min_blob_area:
                x, y, w, h = cv2.boundingRect(best)
                bbox = (x, y, w, h)
                M = cv2.moments(best)
                if M['m00'] != 0:
                    cx, cy = int(M['m10']/M['m00']), int(M['m01']/M['m00'])
                    target_center = (cx, cy)
                    status = "SIGUIENDO"

        # Suavizado del centro
        if target_center is not None:
            if self.smoothed_center is None:
                self.smoothed_center = target_center
            else:
                sx = int((1 - self.smooth_alpha) * self.smoothed_center[0] + self.smooth_alpha * target_center[0])
                sy = int((1 - self.smooth_alpha) * self.smoothed_center[1] + self.smooth_alpha * target_center[1])
                self.smoothed_center = (sx, sy)
            self.track_points.append(self.smoothed_center)
        else:
            if len(self.track_points) > 0:
                self.track_points.append(self.track_points[-1])

        # Visual
        vis = frame.copy()
        if bbox is not None:
            x, y, w, h = bbox
            cv2.rectangle(vis, (x, y), (x+w, y+h), (0, 255, 0), 2)
        for i in range(1, len(self.track_points)):
            if self.track_points[i-1] is None or self.track_points[i] is None:
                continue
            cv2.line(vis, self.track_points[i-1], self.track_points[i], (0, 200, 255), 2)

        H, W = vis.shape[:2]
        err = None
        direction_msg = "LOST"
        out_cmd = 'S'

        # Decidir comando y control de captura
        now = time.time()
        centered = False

        if self.smoothed_center is not None:
            err = (self.smoothed_center[0] - W//2, self.smoothed_center[1] - H//2)
            dx = err[0]
            if dx > self.center_tol_px:
                out_cmd = 'R'; direction_msg = "RIGHT"; centered = False
            elif dx < -self.center_tol_px:
                out_cmd = 'L'; direction_msg = "LEFT"; centered = False
            else:
                out_cmd = 'F'; direction_msg = "CENTERED→FORWARD"; centered = True

        # Contador de frames centrado y captura con cooldown
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
                # Meta simple (coordenadas y hsv usados)
                cx_meta = self.smoothed_center[0] if self.smoothed_center else -1
                cy_meta = self.smoothed_center[1] if self.smoothed_center else -1
                meta_desc = (
                    f"timestamp={ts}, cx={cx_meta}, cy={cy_meta}, "
                    f"hsv_min={tuple(int(x) for x in self.HSV_MIN)}, "
                    f"hsv_max={tuple(int(x) for x in self.HSV_MAX)}"
                )
                self.get_logger().info(f"[Capture] Guardada: {path}")
                # Subida a Drive (opcional)
                if self.use_drive and self.drive is not None:
                    self._upload_to_drive(path, meta_desc=meta_desc)

        # HUD + FPS
        self.put_hud(vis, self.smoothed_center, err, direction_msg, status)
        dt = time.time() - start_t
        self.fps = 1.0 / dt if dt > 0 else 0.0
        self.put_fps(vis, self.fps)

        cv2.imshow('frame', vis)
        cv2.imshow('mask', mask)

        # Publicar si cambió o por keepalive
        ros_now = self.get_clock().now().nanoseconds / 1e9
        if (out_cmd != self.last_cmd) or ((ros_now - self.last_pub_t) >= self.keepalive_sec):
            self.pub_cmd.publish(String(data=out_cmd))
            if out_cmd != self.last_cmd:
                self.get_logger().info(f"/cmd/vision → {out_cmd}")
            self.last_cmd = out_cmd
            self.last_pub_t = ros_now

        # Salir con ESC
        if cv2.waitKey(1) & 0xFF == 27:
            self.get_logger().info("ESC pressed — shutting down vision_tracker...")
            rclpy.shutdown()

    def _get_param_any(self, name):
        # Devuelve el valor tipado de ROS2 directamente.
        return self.get_parameter(name).value

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

