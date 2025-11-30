"""
Pequeño script para probar la conexión a Drive subiendo un JSON vacío.
Usa las mismas credenciales y carpeta que color_detection.py.
"""

import os
import time
from datetime import datetime

USE_GOOGLE_DRIVE = True
DRIVE_FOLDER_ID = "1EdP-E2N8aJFVE3lpVX8mbdzueAb6ceeB"
CRYPTO_CREDENTIALS_FILE = r"/home/mario/OpenCV_VC/cv_testing/credentials.json"
TOKEN_FILE = "token.json"
UPLOAD_RETRIES = 3
RETRY_BACKOFF_SEC = 1.5


def ts_now():
    return datetime.now().strftime("%Y%m%d_%H%M%S_%f")


def main():
    drive = None
    if USE_GOOGLE_DRIVE:
        try:
            from pydrive2.auth import GoogleAuth
            from pydrive2.drive import GoogleDrive
            gauth = GoogleAuth()
            if not os.path.exists(CRYPTO_CREDENTIALS_FILE):
                raise FileNotFoundError(f"No existe {CRYPTO_CREDENTIALS_FILE}.")
            gauth.LoadClientConfigFile(CRYPTO_CREDENTIALS_FILE)
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
            return

    if drive is None:
        print("Drive no disponible, abortando prueba.")
        return

    filename = "test.csv"
    tmp_path = f"_tmp_{filename}"
    # CSV mínimo de prueba
    with open(tmp_path, "w", encoding="utf-8") as f:
        f.write("columna\nvalor_de_prueba\n")

    last_err = None
    for i in range(1, UPLOAD_RETRIES + 1):
        try:
            f = drive.CreateFile({'title': filename, 'parents': [{'id': DRIVE_FOLDER_ID}]})
            f.SetContentFile(tmp_path)
            f.Upload()
            print(f"✅ Subida a Drive: {filename}")
            os.remove(tmp_path)
            return
        except Exception as e:
            last_err = e
            print(f"❌ Error subiendo a Drive (intento {i}/{UPLOAD_RETRIES}): {e}")
            if i < UPLOAD_RETRIES:
                time.sleep(RETRY_BACKOFF_SEC * i)

    print("⛔ No se pudo subir a Drive.", last_err)
    if os.path.exists(tmp_path):
        os.remove(tmp_path)


if __name__ == "__main__":
    main()
