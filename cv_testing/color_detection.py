import cv2
import numpy as np
import time
import os
from datetime import datetime

# ==== ROS 2 ====
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

USE_GOOGLE_DRIVE = True
DRIVE_FOLDER_ID = "1EdP-E2N8aJFVE3lpVX8mbdzueAb6ceeB"

CREDENTIALS_FILE = r"/home/mario/OpenCV_VC/cv_testing/credentials.json"
TOKEN_FILE = "token.json"

CAMERA_INDEX = 0

CENTER_BOX_RATIO = 0.30
MIN_PIXELS_IN_CENTER = 800
COOLDOWN_SECONDS = 3.0
AREA_MIN_CONTOUR = 800
APPLY_MORPH = True
MORPH_KERNEL_SIZE = 5

EPSILON_POLY_FRAC = 0.06
AR_MIN = 0.85
AR_MAX = 1.15
ANGLE_TOL_DEG = 18

UPLOAD_RETRIES = 3
RETRY_BACKOFF_SEC = 1.5

# Serial eliminado: cmd_mux maneja el envío a Arduino

# Publicación ROS
ROS_TOPIC = "/cmd/vision"

COLOR_CMD = {
    "Blanco": "F",
    "Rojo":   "B",
    "Verde":  "R",
    "Azul":   "L",
}

drive = None
if USE_GOOGLE_DRIVE:
    try:
        from pydrive2.auth import GoogleAuth
        from pydrive2.drive import GoogleDrive
        gauth = GoogleAuth()
        if not os.path.exists(CREDENTIALS_FILE):
            raise FileNotFoundError(f"No existe {CREDENTIALS_FILE}.")
        gauth.LoadClientConfigFile(CREDENTIALS_FILE)
        if os.path.exists(TOKEN_FILE):
            gauth.LoadCredentialsFile(TOKEN_FILE)
        if not gauth.credentials or gauth.access_token_expired:
            gauth.LocalWebserverAuth()
            gauth.SaveCredentialsFile(TOKEN_FILE)
        drive = GoogleDrive(gauth)
        folder = drive.CreateFile({'id': DRIVE_FOLDER_ID})
        folder.FetchMetadata(fields='id,title,mimeType')
        if folder['mimeType'] != 'application/vnd.google-apps.folder':
            raise ValueError("El ID proporcionado no es una carpeta de Drive.")
        print("✅ Drive autenticado y carpeta verificada.")
    except Exception as e:
        print("⚠️ No se pudo inicializar Google Drive.", e)
        USE_GOOGLE_DRIVE = False
        drive = None

def ts_now():
    return datetime.now().strftime("%Y%m%d_%H%M%S_%f")

def upload_to_drive(annotated_frame):
    timestamp = ts_now()
    filename = f"capture_{timestamp}.jpg"
    tmp_path = f"_tmp_{filename}"
    cv2.imwrite(tmp_path, annotated_frame)
    last_err = None
    for i in range(1, UPLOAD_RETRIES + 1):
        try:
            f = drive.CreateFile({'title': filename, 'parents': [{'id': DRIVE_FOLDER_ID}]})
            f.SetContentFile(tmp_path)
            f.Upload()
            print(f"✅ Subida a Drive: {filename}")
            os.remove(tmp_path)
            return True
        except Exception as e:
            last_err = e
            print(f"❌ Error subiendo a Drive (intento {i}/{UPLOAD_RETRIES}): {e}")
            if i < UPLOAD_RETRIES:
                time.sleep(RETRY_BACKOFF_SEC * i)
    print("⛔ No se pudo subir a Drive.", last_err)
    if os.path.exists(tmp_path):
        os.remove(tmp_path)
    return False

def serial_init():
    global ser
    try:
        ser = serial.Serial(SERIAL_PORT, SERIAL_BAUD, timeout=0.5)
        time.sleep(2.0)
        print(f"✅ Serial abierto en {SERIAL_PORT} @ {SERIAL_BAUD}")
    except Exception as e:
        ser = None
        print(f"⚠️ No se pudo abrir serial {SERIAL_PORT}: {e}")

def serial_send(cmd: str):
    if ser is None:
        return
    try:
        ser.write((cmd + "\n").encode("ascii"))
        ser.flush()
        if cmd != "S":  # no loguear S
            print(f"➡️  Serial: '{cmd}'")
    except Exception as e:
        print(f"❌ Error enviando por serial: {e}")

def clean_mask(mask):
    if not APPLY_MORPH:
        return mask
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (MORPH_KERNEL_SIZE, MORPH_KERNEL_SIZE))
    opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k, iterations=1)
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, k, iterations=1)
    return closed

def detect_colors(hsv_img, center_roi):
    lower_white = np.array([0, 0, 180],  dtype=np.uint8)
    upper_white = np.array([180, 40,255], dtype=np.uint8)
    lower_red1  = np.array([0, 60, 50],  dtype=np.uint8)   # rojo banda baja (0-12)
    upper_red1  = np.array([12,255,255], dtype=np.uint8)
    lower_green = np.array([35, 50, 40], dtype=np.uint8)
    upper_green = np.array([85,255,255], dtype=np.uint8)
    lower_blue  = np.array([85, 50, 40], dtype=np.uint8)
    upper_blue  = np.array([140,255,255], dtype=np.uint8)

    mask_white = clean_mask(cv2.inRange(hsv_img, lower_white, upper_white))
    mask_red   = clean_mask(cv2.inRange(hsv_img, lower_red1, upper_red1))
    mask_green = clean_mask(cv2.inRange(hsv_img, lower_green, upper_green))
    mask_blue  = clean_mask(cv2.inRange(hsv_img, lower_blue,  upper_blue))

    x1, y1, x2, y2 = center_roi
    center_masks = {
        "Blanco": mask_white[y1:y2, x1:x2],
        "Rojo":   mask_red[y1:y2,   x1:x2],
        "Verde":  mask_green[y1:y2, x1:x2],
        "Azul":   mask_blue[y1:y2,  x1:x2],
    }
    full_masks = {
        "Blanco": mask_white,
        "Rojo":   mask_red,
        "Verde":  mask_green,
        "Azul":   mask_blue,
    }
    return full_masks, center_masks

def angle_between(p0, p1, p2):
    v1 = p0 - p1
    v2 = p2 - p1
    cosang = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-9)
    cosang = np.clip(cosang, -1.0, 1.0)
    return np.degrees(np.arccos(cosang))

def is_square(cnt):
    area = cv2.contourArea(cnt)
    if area < AREA_MIN_CONTOUR: return False
    peri = cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, EPSILON_POLY_FRAC * peri, True)
    if len(approx) != 4: return False
    if not cv2.isContourConvex(approx): return False
    (w, h) = cv2.minAreaRect(approx)[1]
    if w < 1 or h < 1: return False
    ar = max(w, h) / (min(w, h) if min(w, h) > 0 else 1)
    if not (AR_MIN <= ar <= AR_MAX): return False
    pts = approx.reshape(-1, 2).astype(np.float32)
    for i in range(4):
        p0, p1, p2 = pts[(i - 1) % 4], pts[i], pts[(i + 1) % 4]
        ang = angle_between(p0, p1, p2)
        if abs(ang - 90.0) > ANGLE_TOL_DEG: return False
    return True

def largest_square_contour(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    squares = [c for c in contours if is_square(c)]
    if not squares: return None
    return max(squares, key=cv2.contourArea)

def process_frame(frame, last_saved_time):
    height, width = frame.shape[:2]
    cx, cy = width // 2, height // 2
    box_w = int(width * CENTER_BOX_RATIO)
    box_h = int(height * CENTER_BOX_RATIO)
    x1 = max(0, cx - box_w // 2); y1 = max(0, cy - box_h // 2)
    x2 = min(width, cx + box_w // 2); y2 = min(height, cy + box_h // 2)
    center_roi = (x1, y1, x2, y2)

    annotated = frame.copy()
    cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)

    hsv_full = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    full_masks, center_masks = detect_colors(hsv_full, center_roi)

    detections = []
    for color_name in ["Blanco", "Rojo", "Verde", "Azul"]:
        full_mask = full_masks[color_name]
        center_mask = center_masks[color_name]
        center_count = cv2.countNonZero(center_mask)
        if center_count >= MIN_PIXELS_IN_CENTER:
            cnt = largest_square_contour(full_mask)
            if cnt is not None:
                M = cv2.moments(cnt)
                if M["m00"] != 0:
                    cX = int(M["m10"] / M["m00"])
                    cY = int(M["m01"] / M["m00"])
                    detections.append((color_name, cnt, (cX, cY), center_count))

    take_snapshot = False
    chosen_color = None

    if detections:
        detections.sort(key=lambda d: cv2.contourArea(d[1]), reverse=True)
        color_name, cnt, (cX, cY), _ = detections[0]
        chosen_color = color_name
        cv2.drawContours(annotated, [cnt], -1, (255, 0, 0), 2)
        cv2.circle(annotated, (cX, cY), 4, (255, 255, 255), -1)
        cv2.line(annotated, (cx - 10, cy), (cx + 10, cy), (0, 255, 255), 1)
        cv2.line(annotated, (cx, cy - 10), (cx, cy + 10), (0, 255, 255), 1)
        if x1 <= cX <= x2 and y1 <= cY <= y2:
            take_snapshot = True
            cv2.putText(annotated, f"{color_name} cuadrado centrado", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 200, 0), 2)
        else:
            cv2.putText(annotated, f"{color_name} cuadrado detectado", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 200, 200), 2)

    now = time.time()
    if take_snapshot and (now - last_saved_time) >= COOLDOWN_SECONDS:
        last_saved_time = now
    else:
        take_snapshot = False

    return annotated, take_snapshot, last_saved_time, chosen_color

# ====== MAIN ======

# ROS 2 init y publisher
rclpy.init()
ros_node = Node('vision_cmd_pub')
pub = ros_node.create_publisher(String, ROS_TOPIC, 10)

cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_V4L2)
if not cap.isOpened():
    ros_node.get_logger().error(f"No se pudo abrir la cámara en índice {CAMERA_INDEX} con V4L2.")
    raise RuntimeError("Camara no disponible")

w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"✅ Cámara abierta (V4L2). Resolución: {w} x {h}")

last_saved_time = 0.0
last_cmd_sent = None

try:
    while rclpy.ok():
        ret, frame = cap.read()
        if not ret:
            print("⚠️ No se pudo leer frame de la cámara.")
            break

        annotated, take_snapshot, last_saved_time, chosen_color = process_frame(frame, last_saved_time)

        if chosen_color:
            new_cmd = COLOR_CMD.get(chosen_color, "S")
            if take_snapshot and USE_GOOGLE_DRIVE:
                upload_to_drive(annotated)
        else:
            new_cmd = "S"

        if new_cmd != last_cmd_sent:
            
            # ROS publish
            msg = String()
            msg.data = new_cmd
            pub.publish(msg)
            # (opcional) log solo si no es S
            if new_cmd != "S":
                ros_node.get_logger().info(f"/cmd/vision → {new_cmd}")
            last_cmd_sent = new_cmd

        cv2.imshow("frame", annotated)
        if (cv2.waitKey(1) & 0xFF) == ord('q'):
            break

finally:
    cap.release()
    cv2.destroyAllWindows()
    ros_node.destroy_node()
    rclpy.shutdown()
