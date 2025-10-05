# Develop an application that performs color detection on video.

import cv2
import numpy as np
lower_red = np.array([0, 120, 70])
upper_red = np.array([10, 255, 255])
cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if not ret:
        break
    # Convert the frame to HSV color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # Create a mask for red color using the defined bounds
    mask1 = cv2.inRange(hsv, lower_red, upper_red)
    # Define upper range for red color and create another mask
    lower_red2 = np.array([170, 120, 70])
    upper_red2 = np.array([180, 255, 255])
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    # Combine both masks
    mask = mask1 | mask2
    # Find contours in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # Draw bounding boxes around detected red objects
    for contour in contours:
        if cv2.contourArea(contour) > 500:  # Filter out small contours
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    # Display the original frame with detected objects highlighted
    cv2.imshow('Red Color Detection', frame)
    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
# Release the video capture object and close all OpenCV windows
cap.release()
cv2.destroyAllWindows() 
