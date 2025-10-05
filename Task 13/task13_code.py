# Develop an application that tracks black objects through the camera using OpenCV.

import cv2
import numpy as np  
# Define the lower and upper bounds for the color black in HSV space
lower_black = np.array([0, 0, 0])
upper_black = np.array([180, 255, 30])
# Start video capture from the default camera
cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if not ret:
        break
    # Convert the frame to HSV color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # Create a mask for black color using the defined bounds
    mask = cv2.inRange(hsv, lower_black, upper_black)
    # Find contours in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # Draw bounding boxes around detected black objects
    for contour in contours:
        if cv2.contourArea(contour) > 500:  # Filter out small contours
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    # Display the original frame with detected objects highlighted
    cv2.imshow('Black Object Tracking', frame)
    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
# Release the video capture object and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()