import cv2
import os
import time
from datetime import datetime

cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")

evidence_dir = os.path.join("cv_testing", "evidence")
os.makedirs(evidence_dir, exist_ok=True)

last_capture_time = 0.0
font = cv2.FONT_HERSHEY_SIMPLEX

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=6, minSize=(80, 80))

    qualified_faces = 0
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.putText(frame, "face", (x, max(y - 10, 0)), font, 0.6, (255, 0, 0), 2, cv2.LINE_AA)

        roi_gray = gray[y:y + h, x:x + w]
        roi_color = frame[y:y + h, x:x + w]
        eyes = eye_cascade.detectMultiScale(
            roi_gray,
            scaleFactor=1.1,
            minNeighbors=10,
            minSize=(20, 20),
            maxSize=(w // 2, h // 2)
        )

        eyes = sorted(eyes, key=lambda e: e[0])[:2] if len(eyes) else []

        has_eye_detection = False
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)
            cv2.putText(roi_color, "eyes", (ex, max(ey - 5, 0)), font, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
            has_eye_detection = True

        if has_eye_detection:
            qualified_faces += 1

    if qualified_faces >= 3:
        elapsed = time.time() - last_capture_time
        if elapsed >= 3:
            filename = f"capture_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.png"
            cv2.imwrite(os.path.join(evidence_dir, filename), frame)
            last_capture_time = time.time()

    cv2.imshow('frame', frame)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
