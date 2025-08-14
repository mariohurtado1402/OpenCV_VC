import cv2
import numpy as np

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    width = int(cap.get(3))
    height = int(cap.get(4))
    image = np.zeros(frame.shape, np.uint8)
    smaller_frame = cv2.resize(frame, (0,0), fx=0.5, fy=0.5)
    image[:height//2, :width//2] = smaller_frame = cv2.rotate(smaller_frame, cv2.ROTATE_180)
    image[height//2:, :width//2] = smaller_frame
    image[:height//2, width//2:] = smaller_frame = cv2.rotate(smaller_frame, cv2.ROTATE_180)
    image[height//2:, width//2:] = smaller_frame

    img = cv2.line(frame, (0, 0), (width, height), (255, 0, 0), 10)
    img = cv2.line(frame, (width, 0), (0, height), (0, 255, 0), 10)
    img = cv2.circle(image, (width//2, height//2), 50, (0, 0, 255))
    img = cv2.rectangle(image, (0, 0), (width, height), (255, 255, 0), 10)  

    font = cv2.FONT_HERSHEY_SIMPLEX
    img = cv2.putText(frame, 'Cachondas a 1km', (100, height - 10), font, 1, (0, 255, 0), 5, cv2.LINE_AA) 

    cv2.imshow('frame', frame)
    cv2.imshow('image', image)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()