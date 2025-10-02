# Perform Corner Detection and Edge Detection sequentially.

import cv2
import numpy as np

img = cv2.imread('Data/leaf.jpg')
original = img.copy()
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#Corner Detection
gray_float = np.float32(gray)
harris_corners = cv2.cornerHarris(gray_float, blockSize=2, ksize=3, k=0.04)
harris_corners = cv2.dilate(harris_corners, None)
img[harris_corners > 0.01 * harris_corners.max()] = [0, 0, 255]
cv2.imwrite('Data/photo_corners.jpg', img)
# Apply Canny edge detector on original grayscale image
edges = cv2.Canny(gray, threshold1=100, threshold2=200)
cv2.imwrite('Data/photo_edges.jpg', edges)