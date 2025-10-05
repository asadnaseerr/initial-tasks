# Access the camera on computer using OpenCV and display the image on a window.

import cv2
cap = cv2.VideoCapture(0)
ret, frame = cap.read()
if ret:
    cv2.imshow('Camera Feed', frame)
    cv2.waitKey(0)
cap.release()
cv2.destroyAllWindows()