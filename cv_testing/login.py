from pydrive2.auth import GoogleAuth
from pydrive2.drive import GoogleDrive
from pathlib import Path

CREDENTIALS_FILE = r"C:\Users\mario\opencv_vc\cv_testing\credentials.json"  # <- AJUSTA ESTO

gauth = GoogleAuth()
gauth.LoadClientConfigFile(CREDENTIALS_FILE)
gauth.LocalWebserverAuth()
drive = GoogleDrive(gauth)
print("✅ Autenticado")