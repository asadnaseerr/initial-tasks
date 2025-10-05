# Use OpenCV's flip method and explain what it does.

import cv2  
img = cv2.imread('Data/IMG_5378.JPG')
# Flip the image horizontally
flipped_img = cv2.flip(img, 1)
# Save the flipped image
cv2.imwrite('Data/flipped_image.JPG', flipped_img)  

# The flip method used above flips the image around the y-axis (horizontal flip).