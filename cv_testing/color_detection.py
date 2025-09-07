import cv2
import numpy as np
import time
import os
from datetime import datetime

# ===== Configuración general =====
USE_GOOGLE_DRIVE = True
DRIVE_FOLDER_ID = "1EdP-E2N8aJFVE3lpVX8mbdzueAb6ceeB"

CREDENTIALS_FILE = r"/home/mario/OpenCV_VC/cv_testing/credentials.json"
TOKEN_FILE = "token.json"

CAMERA_INDEX = 0  # cámara USB Logitech

CENTER_BOX_RATIO = 0.30
MIN_PIXELS_IN_CENTER = 800          # ya no es crítico, pero lo conservo
COOLDOWN_SECONDS = 3.0
AREA_MIN_CONTOUR = 800              # súbelo si hay ruido
APPLY_MORPH = True
MORPH_KERNEL_SIZE = 5

# Tolerancias para considerar un "cuadrado"
EPSILON_POLY_FRAC = 0.04            # precisión de approxPolyDP (4% del perímetro)
AR_MIN = 0.85                       # relación w/h mínima aceptada
AR_MAX = 1.15                       # relación w/h máxima aceptada
ANGLE_TOL_DEG = 18                  # tolerancia de ángulo respecto a 90° (±18°)

UPLOAD_RETRIES = 3
RETRY_BACKOFF_SEC = 1.5

# ===== Autenticación Drive (opcional) =====
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

# ===== Utilidades =====
def ts_now():
    return datetime.now().strftime("%Y%m%d_%H%M%S_%f")

def upload_to_drive(frame, annotated_frame):
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

def clean_mask(mask):
    if not APPLY_MORPH:
        return mask
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (MORPH_KERNEL_SIZE, MORPH_KERNEL_SIZE))
    opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k, iterations=1)
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, k, iterations=1)
    return closed

def detect_colors(hsv_img, center_roi):
    # Rangos HSV básicos (ajústalos según tu iluminación)
    lower_white = np.array([0, 0, 200]);  upper_white = np.array([180, 30, 255])
    lower_red1  = np.array([0, 120, 70]); upper_red1  = np.array([10, 255, 255])
    lower_red2  = np.array([170,120, 70]); upper_red2 = np.array([180,255, 255])
    lower_green = np.array([40, 70, 70]);  upper_green = np.array([80, 255, 255])
    lower_blue  = np.array([90, 50, 50]);  upper_blue  = np.array([130,255, 255])

    mask_white = clean_mask(cv2.inRange(hsv_img, lower_white, upper_white))
    mask_red   = clean_mask(cv2.bitwise_or(cv2.inRange(hsv_img, lower_red1, upper_red1),
                                           cv2.inRange(hsv_img, lower_red2, upper_red2)))
    mask_green = clean_mask(cv2.inRange(hsv_img, lower_green, upper_green))
    mask_blue  = clean_mask(cv2.inRange(hsv_img, lower_blue, upper_blue))

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

# --- Geometría para validar si un contorno es cuadrado ---
def angle_between(p0, p1, p2):
    """Ángulo en p1 formado por p0-p1-p2 en grados."""
    v1 = p0 - p1
    v2 = p2 - p1
    cosang = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-9)
    cosang = np.clip(cosang, -1.0, 1.0)
    return np.degrees(np.arccos(cosang))

def is_square(cnt):
    area = cv2.contourArea(cnt)
    if area < AREA_MIN_CONTOUR:
        return False

    # Aproximación poligonal
    peri = cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, EPSILON_POLY_FRAC * peri, True)
    if len(approx) != 4:
        return False
    if not cv2.isContourConvex(approx):
        return False

    # Relación ancho/alto del rectángulo de área mínima
    rect = cv2.minAreaRect(approx)
    (w, h) = rect[1]
    if w < 1 or h < 1:
        return False
    ar = (w / h) if w > h else (h / w)  # relación >= 1
    if not (1/AR_MAX <= (w/h if h>0 else 0) <= 1/AR_MIN or AR_MIN <= (w/h if h>0 else 0) <= AR_MAX):
        # Simplifica: comprobar simétricamente
        if not (AR_MIN <= ar <= AR_MAX):
            return False

    # Comprobar ángulos cercanos a 90°
    pts = approx.reshape(-1, 2)
    # Ordenar puntos por ángulo de la caja para consistencia (opcional),
    # basta con recorrerlos en orden.
    for i in range(4):
        p0 = pts[(i - 1) % 4]
        p1 = pts[i]
        p2 = pts[(i + 1) % 4]
        ang = angle_between(p0.astype(np.float32), p1.astype(np.float32), p2.astype(np.float32))
        if abs(ang - 90.0) > ANGLE_TOL_DEG:
            return False

    return True

def largest_square_contour(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    squares = []
    for c in contours:
        if is_square(c):
            squares.append(c)
    if not squares:
        return None
    return max(squares, key=cv2.contourArea)

def process_frame(frame, last_saved_time):
    height, width = frame.shape[:2]
    cx, cy = width // 2, height // 2
    box_w = int(width * CENTER_BOX_RATIO)
    box_h = int(height * CENTER_BOX_RATIO)
    x1 = max(0, cx - box_w // 2)
    y1 = max(0, cy - box_h // 2)
    x2 = min(width, cx + box_w // 2)
    y2 = min(height, cy + box_h // 2)
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
            cnt = largest_square_contour(full_mask)  # <--- solo cuadrados
            if cnt is not None:
                M = cv2.moments(cnt)
                if M["m00"] != 0:
                    cX = int(M["m10"] / M["m00"])
                    cY = int(M["m01"] / M["m00"])
                    detections.append((color_name, cnt, (cX, cY), center_count))

    take_snapshot = False
    if detections:
        # Tomar el cuadrado más grande
        detections.sort(key=lambda d: cv2.contourArea(d[1]), reverse=True)
        color_name, cnt, (cX, cY), _ = detections[0]

        # Dibujo y marca de centro
        cv2.drawContours(annotated, [cnt], -1, (255, 0, 0), 2)
        cv2.circle(annotated, (cX, cY), 4, (255, 255, 255), -1)
        cv2.line(annotated, (cx - 10, cy), (cx + 10, cy), (0, 255, 255), 1)
        cv2.line(annotated, (cx, cy - 10), (cx, cy + 10), (0, 255, 255), 1)

        # Texto y snapshot solo si el cuadrado está centrado
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

    return annotated, take_snapshot, last_saved_time

# ===== Apertura de cámara (V4L2 en Linux/RPi) =====
cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_V4L2)
if not cap.isOpened():
    raise RuntimeError(f"No se pudo abrir la cámara en índice {CAMERA_INDEX} con V4L2.")

print("✅ Cámara abierta (V4L2). Resolución:",
      int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
      "x",
      int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))

# (Opcional) Fijar resolución/fps
# cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
# cap.set(cv2.CAP_PROP_FPS, 30)

last_saved_time = 0.0
try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("⚠️ No se pudo leer frame de la cámara.")
            break

        annotated, take_snapshot, last_saved_time = process_frame(frame, last_saved_time)

        if take_snapshot and USE_GOOGLE_DRIVE:
            upload_to_drive(frame, annotated)

        cv2.imshow("frame", annotated)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s') and USE_GOOGLE_DRIVE:
            upload_to_drive(frame, annotated)
finally:
    cap.release()
    cv2.destroyAllWindows()


