# realtime_detect_debug_v2.py
from ultralytics import YOLO
from pathlib import Path
import cv2, time, os
import numpy as np  # <-- necesario para el warm-up

WEIGHTS = "/Users/mario/opencv_vc/runs/detect/train/weights/best.pt"  # ajusta si cambias de run
CAM_INDEX = 0
CONF_TH = 0.2              # umbral más permisivo para depurar
IMG_SIZE_INFER = 640        # tamaño de inferencia
CLASS_NAMES = {0: "pieza"}  # agrega más si tienes otras clases

def best_device():
    try:
        import torch
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"

def main():
    device = best_device()
    print(f"[INFO] Loading {WEIGHTS} on device={device}")
    model = YOLO(WEIGHTS)

    cap = cv2.VideoCapture(CAM_INDEX)
    if not cap.isOpened():
        raise SystemExit("[ERROR] No se pudo abrir la cámara")

    # Fuerza resolución de captura (opcional)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    os.makedirs("frames", exist_ok=True)
    any_detected = False
    total_kept = 0
    win = "YOLOv8 realtime (q=salir, s=save)"

    # --- Warm-up CORRECTO (opcional): frame negro de NumPy ---
    dummy = np.zeros((IMG_SIZE_INFER, IMG_SIZE_INFER, 3), dtype=np.uint8)
    _ = model.predict(source=dummy, device=device, conf=CONF_TH, imgsz=IMG_SIZE_INFER, verbose=False)
    # ----------------------------------------------------------

    t_prev = time.time()
    while True:
        ok, frame = cap.read()
        if not ok:
            print("[WARN] Frame no OK")
            break

        # Redimensionar a tamaño de inferencia
        frame_in = cv2.resize(frame, (IMG_SIZE_INFER, IMG_SIZE_INFER))

        results = model.predict(
            source=frame_in,
            conf=CONF_TH,
            imgsz=IMG_SIZE_INFER,
            device=device,
            verbose=False
        )
        r = results[0]
        n_all = int(len(r.boxes)) if r.boxes is not None else 0

        kept = 0
        top_info = None
        if n_all > 0:
            confs = r.boxes.conf.cpu().numpy()
            order = confs.argsort()[::-1]
            for idx in order:
                b = r.boxes[idx]
                x1, y1, x2, y2 = [int(v) for v in b.xyxy[0].tolist()]
                cls_id = int(b.cls[0].item()) if b.cls is not None else -1
                conf = float(b.conf[0].item())
                if top_info is None:
                    top_info = (cls_id, conf)
                if conf >= CONF_TH:
                    kept += 1
                    label = CLASS_NAMES.get(cls_id, f"id{cls_id}")
                    cv2.rectangle(frame_in, (x1, y1), (x2, y2), (0, 200, 0), 2)
                    cv2.putText(frame_in, f"{label} {conf:.2f}", (x1, max(20, y1 - 8)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 0), 2)

        if kept > 0:
            any_detected = True
            total_kept += kept

        # HUD
        t_now = time.time()
        fps = 1.0 / max(1e-6, (t_now - t_prev))
        t_prev = t_now
        hud = f"conf>={CONF_TH:.2f} | all:{n_all} kept:{kept} | fps:{fps:4.1f}"
        if top_info:
            hud += f" | top: cls={top_info[0]} conf={top_info[1]:.2f}"
        cv2.putText(frame_in, hud, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

        if kept > 0:
            cv2.putText(frame_in, "PIEZA DETECTADA", (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        cv2.imshow(win, frame_in)
        k = cv2.waitKey(1) & 0xFF
        if k == ord('q'):
            break
        elif k == ord('s'):
            out = Path("frames")/f"cap_{int(time.time()*1000)}.jpg"
            cv2.imwrite(str(out), frame_in)
            print(f"[OK] Guardado {out}")

    cap.release()
    cv2.destroyAllWindows()

    if any_detected:
        print(f"\n[RESULTADO] Sí detectó la pieza. Cajas mostradas: {total_kept}")
    else:
        print("\n[RESULTADO] No hubo detecciones visibles.")
        print("Tips: baja CONF_TH a 0.10, mejora iluminación, acerca el objeto, reentrena con más muestras/variaciones.")

if __name__ == "__main__":
    main()
