# use HSV to change the HSV values ​​of the image in the video using a trackbar.

import cv2
import numpy as np
def nothing(x):
    pass
# Create a window
cv2.namedWindow('HSV Adjustments')
# Create trackbars for hue, saturation, and value adjustments
cv2.createTrackbar('Hue', 'HSV Adjustments', 0, 179, nothing)
cv2.createTrackbar('Saturation', 'HSV Adjustments', 0, 255, nothing)
cv2.createTrackbar('Value', 'HSV Adjustments', 0, 255, nothing)
# Start video capture from the default camera
cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if not ret:
        break
    # Convert the frame to HSV color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # Get current positions of the trackbars
    h = cv2.getTrackbarPos('Hue', 'HSV Adjustments')
    s = cv2.getTrackbarPos('Saturation', 'HSV Adjustments')
    v = cv2.getTrackbarPos('Value', 'HSV Adjustments')
    # Create a new HSV image with adjusted values
    adjusted_hsv = np.copy(hsv)
    adjusted_hsv[:, :, 0] = (adjusted_hsv[:, :, 0] + h) % 180  # Adjust hue
    adjusted_hsv[:, :, 1] = np.clip(adjusted_hsv[:, :, 1] + s, 0, 255)  # Adjust saturation
    adjusted_hsv[:, :, 2] = np.clip(adjusted_hsv[:, :, 2] + v, 0, 255)  # Adjust value
    # Convert back to BGR color space
    adjusted_frame = cv2.cvtColor(adjusted_hsv, cv2.COLOR_HSV2BGR)
    # Display the original and adjusted frames
    cv2.imshow('Original Frame', frame)
    cv2.imshow('Adjusted Frame', adjusted_frame)
    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
# Release the video capture object and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()


# The code above captures video from the camera, converts each frame to HSV color space, and allows the user to adjust the hue, saturation, and value using trackbars. 
# The adjusted frame is then converted back to BGR color space and displayed alongside the original frame. The loop continues until the user presses 'q' to exit.