import cv2
import numpy as np
from collections import deque

# ------------------- Adjustable Parameters -------------------
CAM_INDEX = 0                 # Camera index (0 by default)
MIN_KEYPOINTS = 20            # Minimum ORB keypoints to accept initialization
MIN_GOOD_MATCHES = 25         # Minimum good matches to estimate motion
TRAIL_LEN = 64                # Number of points in the on-screen trail

# Trail controls
SHOW_TRAIL = False            # Toggle draw mode (key: t)
HOLD_TO_SHOW = False          # Hold-to-display mode (toggle key: m)

# HSV thresholds to isolate the black object (tune to your lighting)
HSV_LOW = (0, 0, 0)
HSV_HIGH = (180, 255, 60)     # Low V -> black (raise/lower 60 as needed)

# Debug window for ORB matches
SHOW_MATCHES_WINDOW = True
MATCH_WINDOW_NAME = "ORB matches (debug)"
MAX_MATCHES_TO_DRAW = 80
# -------------------------------------------------------------

def get_black_mask(bgr):
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, HSV_LOW, HSV_HIGH)
    # Morphological cleanup to reduce noise
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    return mask

def largest_contour_bbox(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea)
    if cv2.contourArea(contour) < 500:  # Skip very small blobs (likely noise)
        return None
    x, y, w, h = cv2.boundingRect(contour)
    return x, y, w, h

def extract_orb(frame_gray, mask=None):
    # High-quality ORB tuned for general scenes
    orb = cv2.ORB_create(
        nfeatures=2000,
        scaleFactor=1.2,
        nlevels=8,
        edgeThreshold=15,
        firstLevel=0,
        WTA_K=2,
        scoreType=cv2.ORB_HARRIS_SCORE,
        patchSize=31,
        fastThreshold=12,
    )
    keypoints, descriptors = orb.detectAndCompute(frame_gray, mask)
    return keypoints, descriptors

def good_matches(desc_ref, desc_cur):
    if desc_ref is None or desc_cur is None:
        return []
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    matches_knn = bf.knnMatch(desc_ref, desc_cur, k=2)
    good = []
    for pair in matches_knn:
        if len(pair) == 2:
            m, n = pair
            if m.distance < 0.75 * n.distance:  # Lowe's ratio test
                good.append(m)
    return good

def draw_quad(frame, quad, color=(0, 255, 0), thickness=2):
    quad = quad.astype(int)
    for i in range(4):
        p1 = tuple(quad[i, 0])
        p2 = tuple(quad[(i + 1) % 4, 0])
        cv2.line(frame, p1, p2, color, thickness)

def close_match_window():
    if not SHOW_MATCHES_WINDOW:
        return
    try:
        cv2.destroyWindow(MATCH_WINDOW_NAME)
    except cv2.error:
        pass

def main():
    global SHOW_TRAIL, HOLD_TO_SHOW

    cap = cv2.VideoCapture(CAM_INDEX)
    if not cap.isOpened():
        print("No se pudo abrir la camara.")
        return

    # Persistent state
    ref_frame = None          # Full frame captured at initialization
    ref_kps, ref_des = None, None
    ref_quad = None           # Reference bbox corners (4x1x2)
    trail = deque(maxlen=TRAIL_LEN)
    initialized = False
    show_help = True

    print("[Controles] q: salir | r: reiniciar referencia | h: ver/ocultar ayuda | t: traza on/off (toggle) | m: cambiar modo hold/toggle")

    while True:
        ok, frame = cap.read()
        if not ok:
            print("No se pudo leer frame de la camara.")
            break

        frame_disp = frame.copy()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if not initialized:
            mask = get_black_mask(frame)
            bbox = largest_contour_bbox(mask)

            if bbox is not None:
                x, y, w, h = bbox
                obj_mask = np.zeros_like(mask)
                obj_mask[y:y + h, x:x + w] = mask[y:y + h, x:x + w]
                ref_kps, ref_des = extract_orb(gray, obj_mask)

                if ref_kps and len(ref_kps) >= MIN_KEYPOINTS:
                    ref_frame = frame.copy()
                    ref_quad = np.array(
                        [
                            [[x, y]],
                            [[x + w, y]],
                            [[x + w, y + h]],
                            [[x, y + h]],
                        ],
                        dtype=np.float32,
                    )
                    initialized = True
                    cv2.rectangle(frame_disp, (x, y), (x + w, y + h), (0, 255, 255), 2)
                    cv2.putText(
                        frame_disp,
                        "Referencia inicializada",
                        (x, max(0, y - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 255),
                        2,
                    )
                else:
                    cv2.putText(
                        frame_disp,
                        "Pocas caracteristicas ORB en zona negra",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 0, 255),
                        2,
                    )
            else:
                cv2.putText(
                    frame_disp,
                    "No se detecta objeto negro. Ajusta iluminacion/HSV.",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 0, 255),
                    2,
                )

            if SHOW_MATCHES_WINDOW:
                close_match_window()
        else:
            cur_kps, cur_des = extract_orb(gray, None)
            matches = good_matches(ref_des, cur_des)

            if len(matches) >= MIN_GOOD_MATCHES:
                src_pts = np.float32([ref_kps[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
                dst_pts = np.float32([cur_kps[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

                H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
                M_affine, _ = cv2.estimateAffinePartial2D(
                    src_pts, dst_pts, method=cv2.RANSAC, ransacReprojThreshold=5.0
                )

                used = False
                center = None

                if H is not None:
                    quad = cv2.perspectiveTransform(ref_quad, H)
                    draw_quad(frame_disp, quad, (0, 255, 0), 2)
                    center = quad.mean(axis=0).ravel()
                    used = True
                elif M_affine is not None:
                    quad = cv2.transform(ref_quad, M_affine)
                    draw_quad(frame_disp, quad, (0, 200, 0), 2)
                    center = quad.mean(axis=0).ravel()
                    used = True

                if SHOW_MATCHES_WINDOW and ref_frame is not None:
                    match_vis = cv2.drawMatches(
                        ref_frame,
                        ref_kps,
                        frame,
                        cur_kps,
                        matches[:MAX_MATCHES_TO_DRAW],
                        None,
                        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
                    )
                    cv2.imshow(MATCH_WINDOW_NAME, match_vis)

                if used and center is not None:
                    trail.appendleft((int(center[0]), int(center[1])))
                    cv2.circle(frame_disp, (int(center[0]), int(center[1])), 5, (0, 255, 0), -1)
                else:
                    mask_blk = get_black_mask(frame)
                    bbox = largest_contour_bbox(mask_blk)
                    if bbox is not None:
                        x, y, w, h = bbox
                        cv2.rectangle(frame_disp, (x, y), (x + w, y + h), (0, 165, 255), 2)
                        cx, cy = x + w // 2, y + h // 2
                        trail.appendleft((cx, cy))
                    cv2.putText(
                        frame_disp,
                        "Senal debil: reintentando por color",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 165, 255),
                        2,
                    )
            else:
                if SHOW_MATCHES_WINDOW:
                    close_match_window()
                mask_blk = get_black_mask(frame)
                bbox = largest_contour_bbox(mask_blk)
                if bbox is not None:
                    x, y, w, h = bbox
                    cv2.rectangle(frame_disp, (x, y), (x + w, y + h), (0, 165, 255), 2)
                    cx, cy = x + w // 2, y + h // 2
                    trail.appendleft((cx, cy))
                    cv2.putText(
                        frame_disp,
                        "Pocos matches: guiado por color",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 165, 255),
                        2,
                    )
                else:
                    cv2.putText(
                        frame_disp,
                        "Perdido. Pulsa 'r' para reiniciar.",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 0, 255),
                        2,
                    )

        draw_trail_now = SHOW_TRAIL
        if HOLD_TO_SHOW:
            draw_trail_now = False

        if SHOW_MATCHES_WINDOW and not initialized:
            close_match_window()

        cv2.imshow("Seguimiento ORB (objeto negro)", frame_disp)
        key = cv2.waitKey(1) & 0xFF

        if HOLD_TO_SHOW and key == ord('t'):
            draw_trail_now = True

        if draw_trail_now and len(trail) > 1:
            for i in range(1, len(trail)):
                if trail[i - 1] is None or trail[i] is None:
                    continue
                cv2.line(frame_disp, trail[i - 1], trail[i], (255, 255, 255), 2)
            cv2.imshow("Seguimiento ORB (objeto negro)", frame_disp)

        if show_help:
            help_line1 = "q: salir | r: reiniciar | h: ayuda on/off | t: traza (toggle) | m: modo hold/toggle"
            help_line2 = f"Modo traza: {'HOLD' if HOLD_TO_SHOW else 'TOGGLE'} | Trail: {'ON' if SHOW_TRAIL else 'OFF'}"
            cv2.putText(
                frame_disp,
                help_line1,
                (10, frame_disp.shape[0] - 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (220, 220, 220),
                1,
            )
            cv2.putText(
                frame_disp,
                help_line2,
                (10, frame_disp.shape[0] - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (220, 220, 220),
                1,
            )
            cv2.imshow("Seguimiento ORB (objeto negro)", frame_disp)

        if key == ord('q'):
            break
        elif key == ord('r'):
            initialized = False
            ref_frame = None
            ref_kps, ref_des, ref_quad = None, None, None
            trail.clear()
            if SHOW_MATCHES_WINDOW:
                close_match_window()
            print("Referencia reiniciada.")
        elif key == ord('h'):
            show_help = not show_help
        elif key == ord('t') and not HOLD_TO_SHOW:
            SHOW_TRAIL = not SHOW_TRAIL
        elif key == ord('m'):
            HOLD_TO_SHOW = not HOLD_TO_SHOW
            SHOW_TRAIL = False
            mode_msg = "HOLD (mantener presionada t)" if HOLD_TO_SHOW else "TOGGLE (t para on/off)"
            print(f"Modo traza: {mode_msg}")

    cap.release()
    if SHOW_MATCHES_WINDOW:
        close_match_window()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
