#!/usr/bin/env python3
import cv2
import numpy as np
import time
import os
from datetime import datetime

import rclpy
from rclpy.node import Node


class ColorDetectionNode(Node):
    def __init__(self):
        super().__init__('color_detection_node')

        # ---- Parameters (can be overridden in launch) ----
        self.declare_parameter('use_google_drive', False)
        self.declare_parameter('drive_folder_id', '')
        self.declare_parameter('credentials_file', '')
        self.declare_parameter('token_file', 'token.json')
        self.declare_parameter('camera_index', 0)

        self.declare_parameter('center_box_ratio', 0.30)
        self.declare_parameter('min_pixels_in_center', 800)
        self.declare_parameter('cooldown_seconds', 3.0)
        self.declare_parameter('area_min_contour', 800)
        self.declare_parameter('apply_morph', True)
        self.declare_parameter('morph_kernel_size', 5)

        self.declare_parameter('epsilon_poly_frac', 0.04)
        self.declare_parameter('ar_min', 0.85)
        self.declare_parameter('ar_max', 1.15)
        self.declare_parameter('angle_tol_deg', 18.0)

        # Read parameters
        self.USE_GOOGLE_DRIVE = bool(self.get_parameter('use_google_drive').value)
        self.DRIVE_FOLDER_ID = str(self.get_parameter('drive_folder_id').value)
        self.CREDENTIALS_FILE = str(self.get_parameter('credentials_file').value)
        self.TOKEN_FILE = str(self.get_parameter('token_file').value)
        self.CAMERA_INDEX = int(self.get_parameter('camera_index').value)

        self.CENTER_BOX_RATIO = float(self.get_parameter('center_box_ratio').value)
        self.MIN_PIXELS_IN_CENTER = int(self.get_parameter('min_pixels_in_center').value)
        self.COOLDOWN_SECONDS = float(self.get_parameter('cooldown_seconds').value)
        self.AREA_MIN_CONTOUR = int(self.get_parameter('area_min_contour').value)
        self.APPLY_MORPH = bool(self.get_parameter('apply_morph').value)
        self.MORPH_KERNEL_SIZE = int(self.get_parameter('morph_kernel_size').value)

        self.EPSILON_POLY_FRAC = float(self.get_parameter('epsilon_poly_frac').value)
        self.AR_MIN = float(self.get_parameter('ar_min').value)
        self.AR_MAX = float(self.get_parameter('ar_max').value)
        self.ANGLE_TOL_DEG = float(self.get_parameter('angle_tol_deg').value)

        # Google Drive optional init
        self.drive = None
        if self.USE_GOOGLE_DRIVE:
            self._init_drive()

        # Open camera
        self.cap = cv2.VideoCapture(self.CAMERA_INDEX, cv2.CAP_V4L2)
        if not self.cap.isOpened():
            raise RuntimeError(
                f"Failed to open camera at index {self.CAMERA_INDEX} with V4L2.")

        w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.get_logger().info(f"Camera opened: {w}x{h}")

        self.last_saved_time = 0.0

        # Timer for processing frames
        self.timer = self.create_timer(0.03, self._on_timer)  # ~33 FPS target

    # -------------- Drive helpers --------------
    def _init_drive(self):
        try:
            from pydrive2.auth import GoogleAuth
            from pydrive2.drive import GoogleDrive

            if not self.CREDENTIALS_FILE or not os.path.exists(self.CREDENTIALS_FILE):
                raise FileNotFoundError(
                    f"Credentials file not found: {self.CREDENTIALS_FILE}")

            gauth = GoogleAuth()
            gauth.LoadClientConfigFile(self.CREDENTIALS_FILE)

            if os.path.exists(self.TOKEN_FILE):
                gauth.LoadCredentialsFile(self.TOKEN_FILE)
            if not gauth.credentials or gauth.access_token_expired:
                # NOTE: This opens a browser; typically not desirable on robots.
                gauth.LocalWebserverAuth()
                gauth.SaveCredentialsFile(self.TOKEN_FILE)

            self.drive = GoogleDrive(gauth)
            folder = self.drive.CreateFile({'id': self.DRIVE_FOLDER_ID})
            folder.FetchMetadata(fields='id,title,mimeType')
            if folder['mimeType'] != 'application/vnd.google-apps.folder':
                raise ValueError('Provided Drive ID is not a folder.')
            self.get_logger().info('Google Drive authenticated and folder verified.')
        except Exception as e:
            self.get_logger().warn(f"Failed to init Google Drive, disabling: {e}")
            self.USE_GOOGLE_DRIVE = False
            self.drive = None

    def _ts_now(self):
        return datetime.now().strftime("%Y%m%d_%H%M%S_%f")

    def _upload_to_drive(self, annotated_frame):
        if not (self.USE_GOOGLE_DRIVE and self.drive):
            return False
        filename = f"capture_{self._ts_now()}.jpg"
        tmp_path = f"_tmp_{filename}"
        try:
            cv2.imwrite(tmp_path, annotated_frame)
            f = self.drive.CreateFile({'title': filename, 'parents': [{'id': self.DRIVE_FOLDER_ID}]})
            f.SetContentFile(tmp_path)
            f.Upload()
            self.get_logger().info(f"Uploaded to Drive: {filename}")
            return True
        except Exception as e:
            self.get_logger().warn(f"Drive upload failed: {e}")
            return False
        finally:
            if os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except Exception:
                    pass

    # -------------- Processing helpers --------------
    def _clean_mask(self, mask):
        if not self.APPLY_MORPH:
            return mask
        k = cv2.getStructuringElement(cv2.MORPH_RECT, (self.MORPH_KERNEL_SIZE, self.MORPH_KERNEL_SIZE))
        opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k, iterations=1)
        closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, k, iterations=1)
        return closed

    def _detect_colors(self, hsv_img, center_roi):
        lower_white = np.array([0, 0, 200]);  upper_white = np.array([180, 30, 255])
        lower_red1  = np.array([0, 120, 70]); upper_red1  = np.array([10, 255, 255])
        lower_red2  = np.array([170,120, 70]); upper_red2 = np.array([180,255, 255])
        lower_green = np.array([40, 70, 70]);  upper_green = np.array([80, 255, 255])
        lower_blue  = np.array([90, 50, 50]);  upper_blue  = np.array([130,255, 255])

        mask_white = self._clean_mask(cv2.inRange(hsv_img, lower_white, upper_white))
        mask_red   = self._clean_mask(cv2.bitwise_or(cv2.inRange(hsv_img, lower_red1, upper_red1),
                                                     cv2.inRange(hsv_img, lower_red2, upper_red2)))
        mask_green = self._clean_mask(cv2.inRange(hsv_img, lower_green, upper_green))
        mask_blue  = self._clean_mask(cv2.inRange(hsv_img, lower_blue, upper_blue))

        x1, y1, x2, y2 = center_roi
        center_masks = {
            "Blanco": mask_white[y1:y2, x1:x2],
            "Rojo":   mask_red[y1:y2, x1:x2],
            "Verde":  mask_green[y1:y2, x1:x2],
            "Azul":   mask_blue[y1:y2, x1:x2],
        }
        full_masks = {
            "Blanco": mask_white,
            "Rojo":   mask_red,
            "Verde":  mask_green,
            "Azul":   mask_blue,
        }
        return full_masks, center_masks

    def _angle_between(self, p0, p1, p2):
        v1 = p0 - p1
        v2 = p2 - p1
        cosang = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-9)
        cosang = np.clip(cosang, -1.0, 1.0)
        return np.degrees(np.arccos(cosang))

    def _is_square(self, cnt):
        area = cv2.contourArea(cnt)
        if area < self.AREA_MIN_CONTOUR:
            return False

        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, self.EPSILON_POLY_FRAC * peri, True)
        if len(approx) != 4:
            return False
        if not cv2.isContourConvex(approx):
            return False

        rect = cv2.minAreaRect(approx)
        (w, h) = rect[1]
        if w < 1 or h < 1:
            return False
        ratio = w / h if h > 0 else 0.0
        ar = (w / h) if w > h else (h / w)
        if not (self.AR_MIN <= ar <= self.AR_MAX):
            return False

        pts = approx.reshape(-1, 2)
        for i in range(4):
            p0 = pts[(i - 1) % 4]
            p1 = pts[i]
            p2 = pts[(i + 1) % 4]
            ang = self._angle_between(p0.astype(np.float32), p1.astype(np.float32), p2.astype(np.float32))
            if abs(ang - 90.0) > self.ANGLE_TOL_DEG:
                return False
        return True

    def _largest_square_contour(self, mask):
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        squares = [c for c in contours if self._is_square(c)]
        if not squares:
            return None
        return max(squares, key=cv2.contourArea)

    def _process_frame(self, frame):
        height, width = frame.shape[:2]
        cx, cy = width // 2, height // 2
        box_w = int(width * self.CENTER_BOX_RATIO)
        box_h = int(height * self.CENTER_BOX_RATIO)
        x1 = max(0, cx - box_w // 2)
        y1 = max(0, cy - box_h // 2)
        x2 = min(width, cx + box_w // 2)
        y2 = min(height, cy + box_h // 2)
        center_roi = (x1, y1, x2, y2)

        annotated = frame.copy()
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)

        hsv_full = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        full_masks, center_masks = self._detect_colors(hsv_full, center_roi)

        detections = []
        for color_name in ["Blanco", "Rojo", "Verde", "Azul"]:
            full_mask = full_masks[color_name]
            center_mask = center_masks[color_name]
            center_count = cv2.countNonZero(center_mask)
            if center_count >= self.MIN_PIXELS_IN_CENTER:
                cnt = self._largest_square_contour(full_mask)
                if cnt is not None:
                    M = cv2.moments(cnt)
                    if M["m00"] != 0:
                        cX = int(M["m10"] / M["m00"])
                        cY = int(M["m01"] / M["m00"])
                        detections.append((color_name, cnt, (cX, cY), center_count))

        take_snapshot = False
        if detections:
            detections.sort(key=lambda d: cv2.contourArea(d[1]), reverse=True)
            color_name, cnt, (cX, cY), _ = detections[0]

            cv2.drawContours(annotated, [cnt], -1, (255, 0, 0), 2)
            cv2.circle(annotated, (cX, cY), 4, (255, 255, 255), -1)

            if x1 <= cX <= x2 and y1 <= cY <= y2:
                take_snapshot = True
                cv2.putText(annotated, f"{color_name} cuadrado centrado", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 200, 0), 2)
            else:
                cv2.putText(annotated, f"{color_name} cuadrado detectado", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 200, 200), 2)

        now = time.time()
        if take_snapshot and (now - self.last_saved_time) >= self.COOLDOWN_SECONDS:
            self.last_saved_time = now
        else:
            take_snapshot = False

        return annotated, take_snapshot

    # -------------- Timer callback --------------
    def _on_timer(self):
        ok, frame = self.cap.read()
        if not ok:
            self.get_logger().warn('Failed to read frame from camera.')
            return

        annotated, take_snapshot = self._process_frame(frame)

        if take_snapshot:
            self._upload_to_drive(annotated)

        try:
            cv2.imshow('color_detection', annotated)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                self.get_logger().info('Quit requested via keypress q.')
                rclpy.shutdown()
            elif key == ord('s'):
                self._upload_to_drive(annotated)
        except Exception:
            # Headless environments may not support imshow; ignore
            pass

    def destroy_node(self):
        try:
            if hasattr(self, 'cap') and self.cap:
                self.cap.release()
            try:
                cv2.destroyAllWindows()
            except Exception:
                pass
        finally:
            super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = ColorDetectionNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()

