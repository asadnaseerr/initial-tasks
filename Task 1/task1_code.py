# Open any photo (PNG/JPG) using the OpenCV library. Save the opened photo in the same directory with a different name.
import cv2  
img = cv2.imread('Data/IMG_5378.JPG')
cv2.imwrite('Data/renamed.JPG', img)
