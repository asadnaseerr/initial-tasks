# Draw a square or rectangle around a specific part of the image with a yellow border. Fill the inside of this area.
import cv2
img = cv2.imread('Data/IMG_5378.JPG')
top_left = (800, 1000)
bottom_right = (1500, 1800)
cv2.rectangle(img, top_left, bottom_right, (0, 255, 255), thickness=-1)
cv2.rectangle(img, top_left, bottom_right, (0, 255, 255), thickness=5)
cv2.imwrite('Data/rectangle_filled_image.JPG', img)