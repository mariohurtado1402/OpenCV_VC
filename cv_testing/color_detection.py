import cv2
import numpy as np
import time
import os
from datetime import datetime

USE_GOOGLE_DRIVE = True
DRIVE_FOLDER_ID = "1EdP-E2N8aJFVE3lpVX8mbdzueAb6ceeB"

CREDENTIALS_FILE = r"C:\Users\mario\opencv_vc\cv_testing\credentials.json"
TOKEN_FILE = "token.json"

CAMERA_INDEX = 0
SHOW_DEBUG_WINDOWS = True

CENTER_BOX_RATIO = 0.30
MIN_PIXELS_IN_CENTER = 800
COOLDOWN_SECONDS = 3.0
AREA_MIN_CONTOUR = 500
APPLY_MORPH = True
MORPH_KERNEL_SIZE = 5

UPLOAD_RETRIES = 3
RETRY_BACKOFF_SEC = 1.5

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

def largest_color_contour(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    cnt = max(contours, key=cv2.contourArea)
    if cv2.contourArea(cnt) < AREA_MIN_CONTOUR:
        return None
    return cnt

<<<<<<< HEAD
# ====== Captura de cámara ======
cap = cv2.VideoCapture(1)
=======
def clean_mask(mask):
    if not APPLY_MORPH:
        return mask
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (MORPH_KERNEL_SIZE, MORPH_KERNEL_SIZE))
    opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k, iterations=1)
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, k, iterations=1)
    return closed

def detect_colors(hsv_img, center_roi):
    lower_white = np.array([0, 0, 200]);  upper_white = np.array([180, 30, 255])
    lower_red1  = np.array([0, 120, 70]); upper_red1  = np.array([10, 255, 255])
    lower_red2  = np.array([170, 120, 70]); upper_red2 = np.array([180, 255, 255])
    lower_green = np.array([40, 70, 70]);  upper_green = np.array([80, 255, 255])
    lower_blue  = np.array([90, 50, 50]);  upper_blue  = np.array([130, 255, 255])

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
            cnt = largest_color_contour(full_mask)
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
        cv2.line(annotated, (cx - 10, cy), (cx + 10, cy), (0, 255, 255), 1)
        cv2.line(annotated, (cx, cy - 10), (cx, cy + 10), (0, 255, 255), 1)
        if x1 <= cX <= x2 and y1 <= cY <= y2:
            take_snapshot = True
            cv2.putText(annotated, f"{color_name} centrado", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 200, 0), 2)
        else:
            cv2.putText(annotated, f"{color_name} detectado", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 200, 200), 2)

    now = time.time()
    if take_snapshot and (now - last_saved_time) >= COOLDOWN_SECONDS:
        last_saved_time = now
    else:
        take_snapshot = False

    return annotated, take_snapshot, last_saved_time, full_masks

cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
>>>>>>> 07c8e6be67dfca1355dd515029c9f4afbd6906ad
if not cap.isOpened():
    raise RuntimeError(f"No se pudo abrir la cámara en índice {CAMERA_INDEX}")

last_saved_time = 0.0
try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        annotated, take_snapshot, last_saved_time, full_masks = process_frame(frame, last_saved_time)
        if take_snapshot and USE_GOOGLE_DRIVE:
            upload_to_drive(frame, annotated)
        cv2.imshow("frame", annotated)
        if SHOW_DEBUG_WINDOWS:
            combined = full_masks["Blanco"].copy()
            combined = cv2.bitwise_or(combined, full_masks["Rojo"])
            combined = cv2.bitwise_or(combined, full_masks["Verde"])
            combined = cv2.bitwise_or(combined, full_masks["Azul"])
            cv2.imshow("mask_white", full_masks["Blanco"])
            cv2.imshow("mask_red", full_masks["Rojo"])
            cv2.imshow("mask_green", full_masks["Verde"])
            cv2.imshow("mask_blue", full_masks["Azul"])
            cv2.imshow("mask_combined", combined)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s') and USE_GOOGLE_DRIVE:
            upload_to_drive(frame, annotated)
finally:
    cap.release()
    cv2.destroyAllWindows()
