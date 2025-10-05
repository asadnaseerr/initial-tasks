# Examine the array differences between the grayscale image and the original image.

import cv2  
import numpy as np
img = cv2.imread('Data/IMG_5378.JPG')
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# Convert grayscale image back to BGR format
gray_bgr = cv2.cvtColor(gray_img, cv2.COLOR_GRAY2BGR)
# Calculate the absolute difference between the original image and the grayscale BGR image
difference = cv2.absdiff(img, gray_bgr)
# Save the difference and greyscale image
cv2.imwrite('Data/difference_image.JPG', difference)
cv2.imwrite('Data/gray_image.JPG', gray_img)