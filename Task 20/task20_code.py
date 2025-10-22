# Train the YOLOv5 model with your own dataset and use the model in real time.

import os
import sys
import subprocess
import shutil
from datetime import datetime

# Change this to your dataset folder path
DATASET_DIR = r"/Users/omen/Downloads/Mobile Dataset"

# Automatically create output folder
OUTPUT_DIR = os.path.join(DATASET_DIR, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Paths for training/validation images
TRAIN_PATH = os.path.join(DATASET_DIR, "images", "train")
VAL_PATH = os.path.join(DATASET_DIR, "images", "val")

# Number of classes and their names
NUM_CLASSES = 1
CLASS_NAMES = ["mobile_phone"]

if not os.path.exists("yolov5"):
    print("🔹 Cloning YOLOv5 repository...")
    subprocess.run(["git", "clone", "https://github.com/ultralytics/yolov5.git"], check=True)

os.chdir("yolov5")

print("🔹 Installing dependencies (first run may take a few minutes)...")
subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], check=True)


data_yaml = f"""train: {TRAIN_PATH}
val: {VAL_PATH}

nc: {NUM_CLASSES}
names: {CLASS_NAMES}
"""

with open("data.yaml", "w") as f:
    f.write(data_yaml)

print("data.yaml created successfully!")

print("\n Starting training...")

run_name = f"mobile_detector_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
train_cmd = [
    sys.executable, "train.py",
    "--img", "416",
    "--batch", "4",
    "--epochs", "10",
    "--data", "data.yaml",
    "--weights", "yolov5s.pt",
    "--name", run_name
]
subprocess.run(train_cmd, check=True)

trained_model_dir = os.path.join("runs", "train", run_name)
best_weights = os.path.join(trained_model_dir, "weights", "best.pt")

if os.path.exists(trained_model_dir):
    dest_dir = os.path.join(OUTPUT_DIR, run_name)
    shutil.copytree(trained_model_dir, dest_dir, dirs_exist_ok=True)
    print(f"Training results saved to: {dest_dir}")
else:
    print("Warning: training output folder not found!")

print("\n Testing trained model on validation images...")
detect_val_cmd = [
    sys.executable, "detect.py",
    "--weights", best_weights,
    "--source", VAL_PATH,
    "--conf", "0.5"
]
subprocess.run(detect_val_cmd, check=True)

# Move validation detection results
detect_output = os.path.join("runs", "detect")
if os.path.exists(detect_output):
    latest_detect = sorted(os.listdir(detect_output))[-1]
    detect_path = os.path.join(detect_output, latest_detect)
    shutil.copytree(detect_path, os.path.join(OUTPUT_DIR, run_name, "val_detect"), dirs_exist_ok=True)
    print(f"Validation detections saved to: {os.path.join(OUTPUT_DIR, run_name, 'val_detect')}")

print("\n Starting real-time webcam detection... (Press 'q' to exit)")
webcam_cmd = [
    sys.executable, "detect.py",
    "--weights", best_weights,
    "--source", "0"
]
subprocess.run(webcam_cmd, check=True)
print("\n All done! Model and results saved in:", OUTPUT_DIR)
