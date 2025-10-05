# Detect people from real-time camera images with the YOLOv4 model and print the FPS value in the upper right corner of the screen.

import os
import cv2
import numpy as np
import time

# Paths to model files in Downloads
DOWNLOADS_DIR = os.path.expanduser("~/Downloads")
CFG_PATH = os.path.join(DOWNLOADS_DIR, "yolov4.cfg")
WEIGHTS_PATH = os.path.join(DOWNLOADS_DIR, "yolov4.weights")
COCO_NAMES_PATH = os.path.join(DOWNLOADS_DIR, "coco.names")

# Load YOLOv4 network and COCO class labels
net = cv2.dnn.readNet(WEIGHTS_PATH, CFG_PATH)
with open(COCO_NAMES_PATH, "r") as f:
    classes = f.read().strip().split("\n")

# Create a window with a confidence slider
WINDOW_NAME = "Human Detection"
cv2.namedWindow(WINDOW_NAME)
cv2.createTrackbar("Confidence x100", WINDOW_NAME, 50, 100, lambda x: None)

# Get YOLO output layer names
layer_names = net.getLayerNames()
out_layers = net.getUnconnectedOutLayers()
output_layers = [layer_names[int(i) - 1] for i in out_layers]

cap = cv2.VideoCapture(0)
prev_time = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    height, width = frame.shape[:2]

    # Read confidence threshold from slider
    conf_threshold = cv2.getTrackbarPos("Confidence x100", WINDOW_NAME) / 100.0

    # Prepare input blob and perform forward pass
    blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    detections = net.forward(output_layers)

    boxes, confidences, class_ids = [], [], []

    for output in detections:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if class_id == 0 and confidence > conf_threshold:  # Detect only humans
                cx = int(detection[0] * width)
                cy = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(cx - w / 2)
                y = int(cy - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Apply Non-Maximum Suppression
    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, 0.4)
    if len(indices) > 0:
        for i in indices.flatten():
            x, y, w, h = boxes[i]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            text = f"{classes[class_ids[i]]}: {confidences[i]:.2f}"
            cv2.putText(frame, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Calculate and display FPS
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time) if prev_time != 0 else 0
    prev_time = curr_time
    cv2.putText(frame, f"FPS: {fps:.2f}", (width - 120, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.imshow(WINDOW_NAME, frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
