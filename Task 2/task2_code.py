# Convert the image using OpenCV to grayscale (apply a filter).
import cv2  
img = cv2.imread('Data/IMG_5378.JPG')
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imwrite('Data/gray_image.JPG', gray_img)