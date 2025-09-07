import cv2
import numpy as np
import time
import os
from datetime import datetime

# ====== CONFIGURACIÓN ======
USE_GOOGLE_DRIVE = True  # Pon False si quieres guardar solo en carpeta local (sin API)
DRIVE_FOLDER_ID = "1EdP-E2N8aJFVE3lpVX8mbdzueAb6ceeB"  # <- Reemplaza por el ID de tu carpeta en Drive
LOCAL_SAVE_DIR = "captures"  # Carpeta local temporal para guardar las capturas

CREDENTIALS_FILE = "credentials.json"  # Debe existir si USE_GOOGLE_DRIVE=True
TOKEN_FILE = "token.json"              # Se genera tras el primer login

# Cuánto de "centrado" consideramos (porción del tamaño de la imagen)
CENTER_BOX_RATIO = 0.30  # 30% del ancho/alto centrado; ajusta a tu gusto

# Umbrales
MIN_PIXELS_IN_CENTER = 800       # píxeles mínimos del color dentro de la caja central para considerarlo "presente"
COOLDOWN_SECONDS = 3.0           # tiempo mínimo entre fotos subidas
AREA_MIN_CONTOUR = 500           # área mínima para filtrar ruido

# ====== (Opcional) Autenticación con Google Drive ======
drive = None
if USE_GOOGLE_DRIVE:
    try:
        from pydrive2.auth import GoogleAuth
        from pydrive2.drive import GoogleDrive

        gauth = GoogleAuth()
        if os.path.exists(TOKEN_FILE):
            gauth.LoadCredentialsFile(TOKEN_FILE)
        if not gauth.credentials or gauth.access_token_expired:
            # Esto abrirá el navegador la primera vez
            gauth.LocalWebserverAuth()
            gauth.SaveCredentialsFile(TOKEN_FILE)
        drive = GoogleDrive(gauth)
    except Exception as e:
        print("⚠️ No se pudo inicializar Google Drive. Guardaré localmente. Error:", e)
        USE_GOOGLE_DRIVE = False

# Asegura carpeta local
os.makedirs(LOCAL_SAVE_DIR, exist_ok=True)

def upload_to_drive(local_path, filename):
    """Sube un archivo a Google Drive dentro de DRIVE_FOLDER_ID."""
    if drive is None:
        print("Drive no inicializado; guardado solo local.")
        return False
    try:
        file = drive.CreateFile({'title': filename, 'parents': [{'id': DRIVE_FOLDER_ID}]})
        file.SetContentFile(local_path)
        file.Upload()
        print(f"✅ Subida a Drive: {filename}")
        return True
    except Exception as e:
        print("❌ Error subiendo a Drive:", e)
        return False

def save_and_maybe_upload(frame):
    """Guarda la imagen localmente y, si procede, la sube a Drive."""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    filename = f"capture_{ts}.jpg"
    local_path = os.path.join(LOCAL_SAVE_DIR, filename)
    cv2.imwrite(local_path, frame)
    print(f"💾 Guardada localmente: {local_path}")

    if USE_GOOGLE_DRIVE:
        upload_to_drive(local_path, filename)

def largest_color_contour(mask):
    """Devuelve el contorno más grande con área > AREA_MIN_CONTOUR, o None."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    cnt = max(contours, key=cv2.contourArea)
    if cv2.contourArea(cnt) < AREA_MIN_CONTOUR:
        return None
    return cnt

# ====== Captura de cámara ======
cap = cv2.VideoCapture(1)
if not cap.isOpened():
    raise RuntimeError("No se pudo abrir la cámara")

last_saved_time = 0.0

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("No se pudo leer frame de la cámara")
            break

        height, width = frame.shape[:2]
        cx, cy = width // 2, height // 2  # centro de la imagen

        # Caja central donde exigimos que esté el objeto
        box_w = int(width * CENTER_BOX_RATIO)
        box_h = int(height * CENTER_BOX_RATIO)
        x1 = cx - box_w // 2
        y1 = cy - box_h // 2
        x2 = cx + box_w // 2
        y2 = cy + box_h // 2

        # Visualización de caja central
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # HSV de la imagen completa (mejor para centroides y contornos)
        hsv_full = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Rangos HSV
        lower_white = np.array([0, 0, 200]);  upper_white = np.array([180, 30, 255])
        lower_red1  = np.array([0, 120, 70]); upper_red1  = np.array([10, 255, 255])
        lower_red2  = np.array([170, 120, 70]); upper_red2 = np.array([180, 255, 255])
        lower_green = np.array([40, 70, 70]);  upper_green = np.array([80, 255, 255])
        lower_blue  = np.array([90, 50, 50]);  upper_blue  = np.array([130, 255, 255])

        # Máscaras globales
        mask_white = cv2.inRange(hsv_full, lower_white, upper_white)
        mask_red   = cv2.bitwise_or(cv2.inRange(hsv_full, lower_red1, upper_red1),
                                    cv2.inRange(hsv_full, lower_red2, upper_red2))
        mask_green = cv2.inRange(hsv_full, lower_green, upper_green)
        mask_blue  = cv2.inRange(hsv_full, lower_blue, upper_blue)

        # Para determinar "centrado", miramos solo dentro de la caja central
        center_mask_white = mask_white[y1:y2, x1:x2]
        center_mask_red   = mask_red[y1:y2, x1:x2]
        center_mask_green = mask_green[y1:y2, x1:x2]
        center_mask_blue  = mask_blue[y1:y2, x1:x2]

        # Detecta color predominante en caja central (y también calcula contorno global)
        detections = []
        for color_name, full_mask, center_mask in [
            ("Blanco", mask_white, center_mask_white),
            ("Rojo",   mask_red,   center_mask_red),
            ("Verde",  mask_green, center_mask_green),
            ("Azul",   mask_blue,  center_mask_blue),
        ]:
            center_count = cv2.countNonZero(center_mask)
            if center_count >= MIN_PIXELS_IN_CENTER:
                cnt = largest_color_contour(full_mask)
                if cnt is not None:
                    M = cv2.moments(cnt)
                    if M["m00"] != 0:
                        cX = int(M["m10"] / M["m00"])
                        cY = int(M["m01"] / M["m00"])
                        detections.append((color_name, cnt, (cX, cY), center_count))

        # Si hay al menos una detección centrada, elige la de mayor área
        take_snapshot = False
        if detections:
            # Ordenar por área de contorno (desc)
            detections.sort(key=lambda d: cv2.contourArea(d[1]), reverse=True)
            color_name, cnt, (cX, cY), _ = detections[0]

            # Dibuja contorno y centroide para debug
            cv2.drawContours(frame, [cnt], -1, (255, 0, 0), 2)
            cv2.circle(frame, (cX, cY), 4, (255, 255, 255), -1)
            cv2.line(frame, (cx - 10, cy), (cx + 10, cy), (0, 255, 255), 1)
            cv2.line(frame, (cx, cy - 10), (cx, cy + 10), (0, 255, 255), 1)

            # Comprobación adicional: el centroide del objeto también debe caer en la caja
            if x1 <= cX <= x2 and y1 <= cY <= y2:
                take_snapshot = True
                cv2.putText(frame, f"{color_name} centrado", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 200, 0), 2)
            else:
                cv2.putText(frame, f"{color_name} detectado (no centrado)", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 200, 200), 2)

        # Antirrebote de capturas
        now = time.time()
        if take_snapshot and (now - last_saved_time) >= COOLDOWN_SECONDS:
            save_and_maybe_upload(frame)
            last_saved_time = now

        # Vista de depuración
        combined = cv2.bitwise_or(mask_white, cv2.bitwise_or(mask_red, cv2.bitwise_or(mask_green, mask_blue)))
        cv2.imshow("frame", frame)
        cv2.imshow("mask_white", mask_white)
        cv2.imshow("mask_red", mask_red)
        cv2.imshow("mask_green", mask_green)
        cv2.imshow("mask_blue", mask_blue)
        cv2.imshow("mask_combined", combined)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    cap.release()
    cv2.destroyAllWindows()
