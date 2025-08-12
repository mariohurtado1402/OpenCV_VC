import cv2 

img = cv2.imread('imgs/bs.jpg', 0)      # Reading the image in grayscale

img = cv2.resize(img, (0,0), fx=0.5, fy=0.5)    # Resizing fx and fy are scaling factors
img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)   # Rotating the image 90 degrees clockwise

cv2.imwrite('imgs/bs_resized.jpg', img)  # Saving the modified image

cv2.imshow('Image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()

print(img.shape)