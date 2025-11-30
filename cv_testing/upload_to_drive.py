#!/usr/bin/env python3
import os
import sys
import time
from datetime import datetime

USE_GOOGLE_DRIVE = True
DRIVE_FOLDER_ID = "1EdP-E2N8aJFVE3lpVX8mbdzueAb6ceeB"
CREDENTIALS_FILE = "/home/mario/OpenCV_VC/cv_testing/credentials.json"
TOKEN_FILE = "/home/mario/OpenCV_VC/cv_testing/token.json"
UPLOAD_RETRIES = 3
RETRY_BACKOFF_SEC = 1.5


def main():
    if len(sys.argv) < 2:
        print("Usage: upload_to_drive.py <file_path>")
        return 1
    path = sys.argv[1]
    if not os.path.exists(path):
        print(f"File not found: {path}")
        return 1

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
            print("Drive listo.")
        except Exception as e:
            print("No se pudo inicializar Drive:", e)
            return 1

    filename = os.path.basename(path)
    last_err = None
    for i in range(1, UPLOAD_RETRIES + 1):
        try:
            f = drive.CreateFile({'title': filename, 'parents': [{'id': DRIVE_FOLDER_ID}]})
            f.SetContentFile(path)
            f.Upload()
            print(f"Subido a Drive: {filename}")
            return 0
        except Exception as e:
            last_err = e
            print(f"Error subiendo (intento {i}/{UPLOAD_RETRIES}): {e}")
            if i < UPLOAD_RETRIES:
                time.sleep(RETRY_BACKOFF_SEC * i)
    print("No se pudo subir a Drive:", last_err)
    return 1


if __name__ == "__main__":
    sys.exit(main())
