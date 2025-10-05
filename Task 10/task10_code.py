# Perform 3 morphological operations on the original image.

import cv2  
import numpy as np
img = cv2.imread('Data/IMG_5378.JPG')
# Convert to grayscale
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# Apply binary thresholding
_, binary_img = cv2.threshold(gray_img, 127, 255, cv2.THRESH_BINARY)
# Define a kernel for morphological operations
kernel = np.ones((5, 5), np.uint8)
# Perform erosion
eroded_img = cv2.erode(binary_img, kernel, iterations=1)
# Perform dilation
dilated_img = cv2.dilate(binary_img, kernel, iterations=1)
# Perform opening (erosion followed by dilation)
opened_img = cv2.morphologyEx(binary_img, cv2.MORPH_OPEN, kernel)
# Save the results
cv2.imwrite('Data/eroded_image.JPG', eroded_img)
cv2.imwrite('Data/dilated_image.JPG', dilated_img)
cv2.imwrite('Data/opened_image.JPG', opened_img)