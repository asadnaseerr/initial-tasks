# Implement human detection with the YOLOv4 model and examine the effects on the results by changing the confidence values.

import os
import cv2
import numpy as np

DOWNLOADS_DIR = os.path.expanduser("~/Downloads")
CFG_PATH = os.path.join(DOWNLOADS_DIR, "yolov4.cfg")
WEIGHTS_PATH = os.path.join(DOWNLOADS_DIR, "yolov4.weights")
COCO_NAMES_PATH = os.path.join(DOWNLOADS_DIR, "coco.names")

# Load YOLOv4 network
net = cv2.dnn.readNet(WEIGHTS_PATH, CFG_PATH)
# Load COCO class labels
with open(COCO_NAMES_PATH, "r") as f:
    classes = f.read().strip().split("\n")

def nothing(x):
    pass

cv2.namedWindow("Human Detection")
cv2.createTrackbar("Confidence x100", "Human Detection", 50, 100, nothing)  # default 50%

layer_names = net.getLayerNames()
out_layers = net.getUnconnectedOutLayers()
output_layers = [layer_names[int(i) - 1] for i in out_layers]  # works for all OpenCV 4.x versions

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    height, width = frame.shape[:2]

    # Get confidence threshold from trackbar
    conf_threshold = cv2.getTrackbarPos("Confidence x100", "Human Detection") / 100.0
    blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    detections = net.forward(output_layers)

    boxes = []
    confidences = []
    class_ids = []
    for output in detections:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            # Filter for 'person' class (class_id == 0)
            if class_id == 0 and confidence > conf_threshold:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, 0.4)

    if len(indices) > 0:
        for i in indices.flatten():  # flatten handles 1D or 2D arrays
            x, y, w, h = boxes[i]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            text = f"{classes[class_ids[i]]}: {confidences[i]:.2f}"
            cv2.putText(frame, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Display results
    cv2.imshow("Human Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
