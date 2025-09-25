from pathlib import Path
import numpy as np
import cv2

ASSETS_DIR = Path(__file__).resolve().parent / "assets"
ASSETS_DIR.mkdir(parents=True, exist_ok=True)
input_path = ASSETS_DIR / "papas_noche.jpg"
img = cv2.imread(str(input_path))

img = cv2.resize(img, (0, 0), fx=0.5, fy=0.5)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

corners = cv2.goodFeaturesToTrack(gray, 500, 0.01, 5, blockSize=3, useHarrisDetector=False)
print(corners)

if corners is not None:
    for corner in corners:
        x, y = map(int, corner.ravel())
        cv2.circle(img, (x, y), 5, (255, 0, 0), -1)

    for i in range(len(corners)):
        for j in range(i + 1, len(corners)):
            corner1 = tuple(map(int, corners[i].ravel()))
            corner2 = tuple(map(int, corners[j].ravel()))
            color = tuple(int(v) for v in np.random.randint(0, 255, size=3))
            cv2.line(img, corner1, corner2, color, 1)

output_path = ASSETS_DIR / f"{input_path.stem}_corners{input_path.suffix}"
cv2.imwrite(str(output_path), img)

cv2.imshow('Frame', img)
cv2.waitKey(0)
cv2.destroyAllWindows()

