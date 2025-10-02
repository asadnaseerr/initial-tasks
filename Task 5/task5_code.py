# First, convert the framed area to gray, then blur it. Research where these processes are used in real life.
import cv2
img = cv2.imread('Data/IMG_5378.JPG')
top_left = (800, 1000)
bottom_right = (1500, 1800)
framed_area = img[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]
gray_area = cv2.cvtColor(framed_area, cv2.COLOR_BGR2GRAY)
blurred_area = cv2.GaussianBlur(gray_area, (15, 15), 0)
img[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]] = cv2.cvtColor(blurred_area, cv2.COLOR_GRAY2BGR)
cv2.imwrite('Data/blurred_framed_area_image.JPG', img) 