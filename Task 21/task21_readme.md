# Task: Examine the model's outputs (such as the F1 score) and interpret them to understand their meaning. Document the interpretation of the outputs in a README.md file.

## Overview
This project trains a YOLOv5 model to detect mobile phones in real time using a custom dataset of 100 labeled images.  
The dataset was annotated in XML format and converted to YOLO labels.

## Training Summary
- Framework: **YOLOv5**
- Dataset: 100 custom images
- Training duration: 10 epochs
- Environment: Python, PyTorch, CUDA

## Model Performance (Final Epoch)

| Metric | Value | Interpretation |
|:--|:--:|:--|
| **Precision** | 0.627 | Model correctly identifies most phones but still produces some false positives. |
| **Recall** | 0.389 | The model detects about 39% of actual phones, suggesting under-detection. |
| **F1 Score** | 0.48 | Moderate overall performance — needs improvement through more data or longer training. |
| **mAP@0.5** | 0.461 | Average bounding box match quality; fair detection accuracy. |
| **mAP@0.5:0.95** | 0.224 | Lower consistency under stricter IoU thresholds, typical for early-stage models. |

## Interpretation
The model demonstrates a **basic ability** to detect mobile phones but with limited recall.  
It tends to **miss some phones** and **misclassify** a few non-phone regions.  
These results suggest the model is learning relevant features, but more training data or augmentation will improve generalization.

### Recommendations
- Increase dataset size beyond 100 images.
- Use **data augmentation** (rotation, brightness, cropping).
- Train for **more epochs** (e.g., 100+).
- Fine-tune with a pre-trained YOLOv5 model (`yolov5s.pt`).

## Visual Results
YOLOv5 automatically saves result plots in:
- `runs/train/exp/F1_curve.png`
- `runs/train/exp/results.png`
- `runs/train/exp/confusion_matrix.png`

## Real-Time Inference
To run the trained model on live video or webcam:
```bash
python detect.py --weights runs/train/exp/weights/best.pt --source 0
