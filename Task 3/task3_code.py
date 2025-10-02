# Resize the image. After this process, create an image that shows the specific area of ​​your choice.
import cv2
img = cv2.imread('Data/IMG_5378.JPG')
resized_img = cv2.resize(img, (800, 600))  
cropped_img = resized_img[100:400, 200:600]  
cv2.imwrite('Data/resized_image.JPG', resized_img)
cv2.imwrite('Data/cropped_image.JPG', cropped_img)