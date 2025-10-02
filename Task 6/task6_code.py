# Perform thresholding operation on a leaf image. (image is in Data folder)
import cv2
img = cv2.imread('Data/leaf.jpg', cv2.IMREAD_GRAYSCALE)
_, thresholded_img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
cv2.imwrite('Data/thresholded_leaf.jpg', thresholded_img)

# Thresholding is used in various applications such as image segmentation, object detection, 
# and computer vision tasks to separate objects from the background based on pixel intensity.