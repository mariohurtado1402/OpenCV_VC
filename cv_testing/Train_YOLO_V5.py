# train_and_predict.py
# ------------------------------------------------------------
# Entrena YOLOv8 y corre predicción sobre images/val
# - Detecta automáticamente la última sesión run_* en dataset/captures
# - Limpia .cache para reindexar
# - Genera un auto_data.yaml apuntando a esa sesión
# - Usa rutas ABSOLUTAS para predict (evita errores por cwd)
# ------------------------------------------------------------

from pathlib import Path
import argparse, sys, yaml
from ultralytics import YOLO

# ---------- Utilidades ----------

def best_device():
    """Devuelve 'mps' en Mac si está disponible; si no, 'cpu'."""
    try:
        import torch
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"

def latest_run(captures_root: Path) -> Path | None:
    """Devuelve la carpeta run_* más reciente dentro de captures/."""
    runs = sorted(
        [p for p in captures_root.glob("run_*") if p.is_dir()],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return runs[0] if runs else None

def write_data_yaml(session_root: Path) -> Path:
    """Crea un data.yaml temporal apuntando a la sesión detectada."""
    data = {
        "path": str(session_root),
        "train": "images/train",
        "val": "images/val",
        "names": {0: "pieza"},  # 👉 Cambia/expande clases aquí si agregas más
    }
    out = session_root / "auto_data.yaml"
    out.write_text(yaml.safe_dump(data, sort_keys=False))
    return out

def clean_caches(session_root: Path):
    """Elimina los .cache para forzar reindexado."""
    for p in [session_root / "train.cache", session_root / "val.cache"]:
        if p.exists():
            p.unlink()

# ---------- Main ----------

def main():
    ap = argparse.ArgumentParser(
        description="Entrena YOLOv8 y predice en images/val usando la última sesión run_*"
    )
    ap.add_argument(
        "--dataset_root",
        default="/Users/mario/opencv_vc/cv_testing/dataset",  # 👉 Cambia si mueves tu carpeta dataset
        help="Raíz del dataset que contiene 'captures/'",
    )
    ap.add_argument("--model", default="yolov8n.pt", help="Pesos base")  # 👉 Cambia a yolov8s.pt, etc.
    ap.add_argument("--imgsz", type=int, default=640)                    # 👉 Cambia tamaño de imagen
    ap.add_argument("--epochs", type=int, default=80)                    # 👉 Cambia épocas
    ap.add_argument("--batch", type=int, default=8)                      # 👉 Ajusta a tu RAM/VRAM
    ap.add_argument("--single_cls", action="store_true", default=True)   # 👉 Desactiva si agregas más clases
    args = ap.parse_args()

    dataset_root = Path(args.dataset_root).expanduser().resolve()
    captures = dataset_root / "captures"
    if not captures.exists():
        sys.exit(f"[ERROR] No existe {captures}")

    session_root = latest_run(captures)
    if not session_root:
        sys.exit(f"[ERROR] No se encontraron sesiones run_* en {captures}")

    print(f"[INFO] Usando sesión más reciente: {session_root}")

    im_val = session_root / "images" / "val"
    if not im_val.exists() or not any(im_val.glob("*")):
        sys.exit(
            f"[ERROR] '{im_val}' no existe o está vacío.\n"
            f"       Copia algunas imágenes a images/val (y sus .txt si quieres validar métricas)."
        )

    # 1) YAML temporal + limpiar caches
    data_yaml = write_data_yaml(session_root)
    clean_caches(session_root)

    # 2) Entrenamiento
    device = best_device()
    print(f"[INFO] Training on device={device}")
    model = YOLO(args.model)
    results = model.train(
        data=str(data_yaml),
        imgsz=args.imgsz,
        epochs=args.epochs,
        batch=args.batch,
        device=device,
        single_cls=args.single_cls,
        seed=42,
    )

    # 3) Predicción sobre images/val (ruta absoluta)
    weights = Path(results.save_dir) / "weights" / "best.pt"
    if not weights.exists():
        sys.exit(f"[ERROR] No se encontró best.pt en {weights}")
    print(f"[INFO] Using weights: {weights}")

    out_project = Path.home() / "pruebasOpenCV" / "runs"   # 👉 Cambia destino si quieres otro folder
    preds = YOLO(str(weights)).predict(
        source=str(im_val),       # RUTA ABSOLUTA => evita FileNotFoundError
        device=device,
        save=True,
        project=str(out_project),
        name="predict_val",       # 👉 Cambia el nombre del run de predicción si quieres
        exist_ok=True
    )
    print("[OK] Predicciones guardadas en:", preds[0].save_dir)
    print("[OK] Entrenamiento terminado. Revisa:", results.save_dir)

if __name__ == "__main__":
    main()
