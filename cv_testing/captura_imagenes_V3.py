# captura_imagenes.py
import cv2, os, argparse, time, json
from pathlib import Path
from datetime import datetime
import numpy as np
import sys

def clip(val, lo, hi): return max(lo, min(hi, val))

def xyxy_to_yolo(x1, y1, x2, y2, img_w, img_h):
    x1, y1, x2, y2 = float(x1), float(y1), float(x2), float(y2)
    w, h = x2 - x1, y2 - y1
    cx, cy = x1 + w / 2.0, y1 + h / 2.0
    return (cx / img_w, cy / img_h, w / img_w, h / img_h)

def yolo_to_xyxy(cx, cy, w, h, img_w, img_h):
    cx, cy, w, h = cx * img_w, cy * img_h, w * img_w, h * img_h
    x1, y1 = cx - w/2, cy - h/2
    x2, y2 = cx + w/2, cy + h/2
    return int(x1), int(y1), int(x2), int(y2)

def make_session_root(base_dataset: Path, session_name: str | None) -> Path:
    captures_root = base_dataset / "captures"
    captures_root.mkdir(parents=True, exist_ok=True)
    if not session_name:
        session_name = time.strftime("run_%Y-%m-%d_%H-%M-%S")
    session_root = captures_root / session_name
    session_root.mkdir(parents=True, exist_ok=True)  # reusar si existe
    return session_root

def ensure_dirs(session_root: Path, split: str):
    img_dir = session_root / "images" / split
    lbl_dir = session_root / "labels" / split
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)
    return img_dir, lbl_dir

def save_label(lbl_path: Path, boxes, class_id: int, w: int, h: int):
    with open(lbl_path, "w") as f:
        for (x1, y1, x2, y2) in boxes:
            cx, cy, ww, hh = xyxy_to_yolo(x1, y1, x2, y2, w, h)
            f.write(f"{class_id} {cx:.6f} {cy:.6f} {ww:.6f} {hh:.6f}\n")

def augment_photometric(img: np.ndarray) -> np.ndarray:
    out = img.astype(np.float32) / 255.0
    gamma = np.random.uniform(0.9, 1.1)
    out = np.power(out, gamma)
    alpha = np.random.uniform(0.9, 1.1)
    beta  = np.random.uniform(-0.03, 0.03)
    out = np.clip(out * alpha + beta, 0.0, 1.0)
    return (out * 255).astype(np.uint8)

def crop_by_box(image: np.ndarray, boxes, class_id: int):
    """
    Recorta por la PRIMERA caja y re-mapea las etiquetas al recorte.
    Si solo hay 1 caja, el label será [class 0.5 0.5 1 1].
    """
    if not boxes:
        return None, []
    H, W = image.shape[:2]
    x1, y1, x2, y2 = boxes[0]
    x1 = clip(x1, 0, W-1); x2 = clip(x2, 0, W-1)
    y1 = clip(y1, 0, H-1); y2 = clip(y2, 0, H-1)
    if x2 <= x1 or y2 <= y1:
        return None, []

    crop = image[y1:y2, x1:x2].copy()
    cw, ch = x2 - x1, y2 - y1

    # re-map de TODAS las cajas que caen dentro del recorte
    remapped = []
    for (bx1, by1, bx2, by2) in boxes:
        nx1 = clip(bx1 - x1, 0, cw); ny1 = clip(by1 - y1, 0, ch)
        nx2 = clip(bx2 - x1, 0, cw); ny2 = clip(by2 - y1, 0, ch)
        if nx2 - nx1 > 2 and ny2 - ny1 > 2:
            remapped.append((int(nx1), int(ny1), int(nx2), int(ny2)))

    # caso simple: si sólo había 1 caja, usamos la caja completa del recorte
    if len(boxes) == 1:
        remapped = [(0, 0, cw, ch)]  # equivale a 0 0.5 0.5 1 1

    return crop, remapped

class Annotator:
    def __init__(self, image):
        self.orig = image.copy()
        self.image = image.copy()
        self.h, self.w = image.shape[:2]
        self.boxes = []
        self.drawing = False
        self.pt1 = None
        self.pt2 = None

    def reset(self):
        self.image = self.orig.copy()
        self.boxes.clear()
        self.drawing = False
        self.pt1 = None
        self.pt2 = None

    def undo(self):
        if self.boxes:
            self.boxes.pop()
            self.redraw()

    def redraw(self):
        self.image = self.orig.copy()
        for (x1, y1, x2, y2) in self.boxes:
            cv2.rectangle(self.image, (x1, y1), (x2, y2), (0,255,0), 2)

    def mouse_cb(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.pt1 = (x, y); self.pt2 = (x, y)
        elif event == cv2.EVENT_MOUSEMOVE and self.drawing:
            self.pt2 = (x, y)
            self.redraw()
            cv2.rectangle(self.image, self.pt1, self.pt2, (0,255,255), 2)
        elif event == cv2.EVENT_LBUTTONUP and self.drawing:
            self.drawing = False
            self.pt2 = (x, y)
            x1, y1 = self.pt1; x2, y2 = self.pt2
            x1, x2 = sorted([x1, x2]); y1, y2 = sorted([y1, y2])
            if abs(x2 - x1) > 5 and abs(y2 - y1) > 5:
                self.boxes.append((x1, y1, x2, y2))
            self.redraw()

def main():
    ap = argparse.ArgumentParser(description="Captura con anotación y guardado TRAIN+VAL")
    ap.add_argument("--dataset", default="/Users/mario/opencv_vc/cv_testing/dataset")
    ap.add_argument("--class_id", type=int, default=0)
    ap.add_argument("--cam", type=int, default=0)
    ap.add_argument("--session", default=None)
    ap.add_argument("--val_mode", default="photometric", choices=["photometric","recapture"])
    ap.add_argument("--save_mode", default="full", choices=["full","crop"],
                    help="full=guarda imagen completa; crop=recorta al primer bbox y ajusta etiquetas")
    args = ap.parse_args()

    base_dataset = Path(args.dataset)
    session_root = make_session_root(base_dataset, args.session)

    info = {
        "session_name": session_root.name,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "class_id": args.class_id,
        "camera_index": args.cam,
        "val_mode": args.val_mode,
        "save_mode": args.save_mode
    }
    (session_root / "session_info.json").write_text(json.dumps(info, indent=2))

    img_train, lbl_train = ensure_dirs(session_root, "train")
    img_val,   lbl_val   = ensure_dirs(session_root, "val")

    cap = cv2.VideoCapture(args.cam)
    if not cap.isOpened():
        print("[ERR] No se pudo abrir la cámara", file=sys.stderr)
        sys.exit(1)

    print("=== Sesión ===")
    print(f"Raíz: {session_root}")
    print("Controles: 'c' capturar/anotar | 'q' salir")
    print("En 'Anotar': z=undo | r=reset | s=guardar (TRAIN+VAL) | ESC=cancelar")
    print(f"Modo guardado: {args.save_mode}")

    while True:
        ok, frame = cap.read()
        if not ok:
            print("[ERR] Frame no OK", file=sys.stderr)
            break

        disp = frame.copy()
        cv2.putText(disp, f"c=capturar | q=salir | mode={args.save_mode}", (15, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 2)
        cv2.imshow("Webcam", disp)
        k = cv2.waitKey(1) & 0xFF
        if k == ord('q'):
            break

        if k == ord('c'):
            ann = Annotator(frame)
            cv2.namedWindow("Anotar")
            cv2.setMouseCallback("Anotar", ann.mouse_cb)
            while True:
                ui = ann.image.copy()
                cv2.putText(ui, "Dibuja cajas. s=guardar | z=undo | r=reset | ESC=cancelar",
                            (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (50,200,255), 2)
                cv2.imshow("Anotar", ui)
                kk = cv2.waitKey(10) & 0xFF
                if kk == ord('z'):
                    ann.undo()
                elif kk == ord('r'):
                    ann.reset()
                elif kk == 27:
                    break
                elif kk == ord('s'):
                    # === BLOQUEO: no se guarda si NO hay cajas ===
                    if len(ann.boxes) == 0:
                        print("[WARN] No hay cajas. Dibuja al menos una antes de guardar.")
                        continue

                    ts = int(time.time() * 1000)

                    # --------- preparar imagen y cajas según modo ----------
                    if args.save_mode == "crop":
                        crop, remapped = crop_by_box(ann.orig, ann.boxes, args.class_id)
                        if crop is None or not remapped:
                            print("[WARN] Recorte inválido. Guardando en modo 'full'.")
                            img_save = ann.orig
                            boxes_to_save = ann.boxes
                            H, W = ann.h, ann.w
                        else:
                            img_save = crop
                            boxes_to_save = remapped
                            H, W = img_save.shape[:2]
                    else:
                        img_save = ann.orig
                        boxes_to_save = ann.boxes
                        H, W = ann.h, ann.w

                    # TRAIN
                    img_name_tr = f"img_{ts}_tr.jpg"
                    lbl_name_tr = f"img_{ts}_tr.txt"
                    cv2.imwrite(str(img_train / img_name_tr), img_save)
                    save_label(lbl_train / lbl_name_tr, boxes_to_save, args.class_id, W, H)
                    print(f"[OK] TRAIN: {img_name_tr} ({len(boxes_to_save)} bbox)")

                    # VAL
                    if args.val_mode == "photometric":
                        img_val_img = augment_photometric(img_save)
                    else:
                        ok2, frame2 = cap.read()
                        img_val_img = frame2.copy() if ok2 else img_save

                    img_name_val = f"img_{ts}_val.jpg"
                    lbl_name_val = f"img_{ts}_val.txt"
                    cv2.imwrite(str(img_val / img_name_val), img_val_img)
                    # mismas cajas (geométricamente válidas)
                    save_label(lbl_val / lbl_name_val, boxes_to_save, args.class_id, W, H)
                    print(f"[OK] VAL:   {img_name_val} ({len(boxes_to_save)} bbox)")
                    break
            cv2.destroyWindow("Anotar")

    cap.release()
    cv2.destroyAllWindows()
    print("\n[LOG] cierre OK")
    print("Apunta tu data.yaml a:")
    print(f"  path: {session_root}")
    print("  train: images/train")
    print("  val:   images/val")
    print("  names: [pieza]")

if __name__ == "__main__":
    main()
