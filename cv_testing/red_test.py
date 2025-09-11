import cv2
import numpy as np

cap = cv2.VideoCapture(0)

# Definimos los dos rangos de rojo
lower_red1 = np.array([0, 40, 40], dtype=np.uint8)
upper_red1 = np.array([12, 255, 255], dtype=np.uint8)

lower_red2 = np.array([170, 40, 40], dtype=np.uint8)
upper_red2 = np.array([179, 255, 255], dtype=np.uint8)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    mask_red1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask_red2 = cv2.inRange(hsv, lower_red2, upper_red2)

    # Combinar las dos máscaras
    mask_red = cv2.bitwise_or(mask_red1, mask_red2)

    # Mostrar resultados
    cv2.imshow("Frame", frame)
    cv2.imshow("Red1 (0-10)", mask_red1)
    cv2.imshow("Red2 (170-179)", mask_red2)
    cv2.imshow("Red Combined", mask_red)

    # Mostrar HSV del centro de la imagen
    h, w = hsv.shape[:2]
    cx, cy = w // 2, h // 2
    pixel = hsv[cy, cx]
    cv2.circle(frame, (cx, cy), 5, (0,255,0), 2)
    print(f"Centro HSV: H={pixel[0]}, S={pixel[1]}, V={pixel[2]}")

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

