# Design a PyQt5 interface where you can view and analyze the output of a ready-made semantic segmentation model. 
# (The original and processed image should be displayed simultaneously in the window. Design the model file and the image to be processed so that they can be selected from the interface.)

import sys
import torch
import numpy as np
import cv2
from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton,
    QHBoxLayout, QVBoxLayout, QFileDialog, QMessageBox
)
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt
from torchvision import transforms

class SegmentationViewer(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("PyTorch Semantic Segmentation Viewer")
        self.resize(1000, 600)

        self.model = None
        self.image_path = None

        # Buttons
        self.load_model_btn = QPushButton("Load Model")
        self.load_image_btn = QPushButton("Load Image")
        self.run_btn = QPushButton("Run Segmentation")
        self.run_btn.setEnabled(False)

        # Image labels
        self.original_label = QLabel("Original Image")
        self.segmented_label = QLabel("Segmented Output")
        for lbl in (self.original_label, self.segmented_label):
            lbl.setAlignment(Qt.AlignCenter)
            lbl.setStyleSheet("border: 1px solid gray; background: #fafafa;")

        # Layouts
        top_layout = QHBoxLayout()
        top_layout.addWidget(self.load_model_btn)
        top_layout.addWidget(self.load_image_btn)
        top_layout.addWidget(self.run_btn)

        img_layout = QHBoxLayout()
        img_layout.addWidget(self.original_label)
        img_layout.addWidget(self.segmented_label)

        main_layout = QVBoxLayout()
        main_layout.addLayout(top_layout)
        main_layout.addLayout(img_layout)
        self.setLayout(main_layout)

        # Connect signals
        self.load_model_btn.clicked.connect(self.load_model)
        self.load_image_btn.clicked.connect(self.load_image)
        self.run_btn.clicked.connect(self.run_segmentation)

        # Transform for preprocessing
        self.preprocess = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((256, 256)),  # adjust depending on your model
            transforms.ToTensor(),
        ])

    def load_model(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Model File", "", "PyTorch Model (*.pt *.pth);;All Files (*)"
        )
        if path:
            try:
                self.model = torch.load(path, map_location=torch.device('cpu'))
                self.model.eval()
                QMessageBox.information(self, "Model Loaded", f"Model loaded:\n{path}")
                self.check_ready()
            except Exception as e:
                QMessageBox.warning(self, "Error", f"Failed to load model:\n{e}")

    def load_image(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Image", "", "Images (*.png *.jpg *.jpeg *.bmp)"
        )
        if path:
            self.image_path = path
            self.display_image(path, self.original_label)
            self.check_ready()

    def check_ready(self):
        self.run_btn.setEnabled(bool(self.model and self.image_path))

    def run_segmentation(self):
        if not self.model or not self.image_path:
            QMessageBox.warning(self, "Error", "Load model and image first.")
            return

        # Read image
        img = cv2.imread(self.image_path)
        if img is None:
            QMessageBox.warning(self, "Error", "Failed to read image.")
            return
        orig_h, orig_w = img.shape[:2]

        # Preprocess
        input_tensor = self.preprocess(img).unsqueeze(0)  # add batch dimension

        # Run model
        with torch.no_grad():
            output = self.model(input_tensor)
            if isinstance(output, (tuple, list)):
                output = output[0]  # sometimes models return tuple
            mask = torch.argmax(output.squeeze(), dim=0).cpu().numpy()  # H x W

        # Resize mask to original image size
        mask_resized = cv2.resize(mask.astype(np.uint8), (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)

        # Generate color overlay
        color_mask = np.zeros_like(img)
        # Assign colors per class (example for 3 classes, extend as needed)
        colors = [
            [0, 0, 0],        # class 0 -> black / background
            [0, 0, 255],      # class 1 -> red
            [0, 255, 0],      # class 2 -> green
            [255, 0, 0],      # class 3 -> blue
        ]
        for i, color in enumerate(colors):
            color_mask[mask_resized == i] = color

        segmented = cv2.addWeighted(img, 0.7, color_mask, 0.3, 0)

        # Display result
        self.display_cv_image(segmented, self.segmented_label)

    def display_image(self, path, label):
        pixmap = QPixmap(path)
        if pixmap.isNull():
            label.setText("Cannot load image")
        else:
            label.setPixmap(pixmap.scaled(label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def display_cv_image(self, img, label):
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        bytes_per_line = ch * w
        qt_img = QImage(rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_img)
        label.setPixmap(pixmap.scaled(label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))


if __name__ == "__main__":
    app = QApplication(sys.argv)
    viewer = SegmentationViewer()
    viewer.show()
    sys.exit(app.exec_())
