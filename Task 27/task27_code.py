# Develop a system that will read optical characters (OCR) from a file selected via the interface. Display the output on a "label" section of the interface.

import sys
import os
import cv2
import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, 
                             QWidget, QLabel, QPushButton, QFileDialog, QTextEdit,
                             QGroupBox, QComboBox, QProgressBar, QMessageBox, QCheckBox)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QFont, QPixmap, QImage
import pytesseract
from PIL import Image, ImageEnhance, ImageFilter


class SimpleOCRSystem(QMainWindow):
    def __init__(self):
        super().__init__()
        self.current_file_path = None
        self.init_ui()
        
    def init_ui(self):
        self.setWindowTitle("Simple OCR System")
        self.setGeometry(100, 100, 1000, 700)
        
        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        layout = QVBoxLayout()
        central_widget.setLayout(layout)
        
        # Title
        title = QLabel("Optical Character Recognition (OCR) System")
        title.setFont(QFont("Arial", 16, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)
        
        # File selection
        file_layout = QHBoxLayout()
        self.select_button = QPushButton("Select Image File")
        self.select_button.clicked.connect(self.select_file)
        self.file_label = QLabel("No file selected")
        file_layout.addWidget(self.select_button)
        file_layout.addWidget(self.file_label)
        layout.addLayout(file_layout)
        
        # Preview
        self.preview_label = QLabel()
        self.preview_label.setAlignment(Qt.AlignCenter)
        self.preview_label.setMinimumHeight(200)
        self.preview_label.setStyleSheet("border: 1px solid gray; background-color: #f0f0f0;")
        self.preview_label.setText("Image preview will appear here")
        layout.addWidget(self.preview_label)
        
        # Process button
        self.process_button = QPushButton("Extract Text with OCR")
        self.process_button.clicked.connect(self.process_ocr)
        self.process_button.setEnabled(False)
        layout.addWidget(self.process_button)
        
        # Results
        results_label = QLabel("Extracted Text:")
        results_label.setFont(QFont("Arial", 12, QFont.Bold))
        layout.addWidget(results_label)
        
        self.results_text = QTextEdit()
        self.results_text.setPlaceholderText("Extracted text will appear here...")
        layout.addWidget(self.results_text)
        
        # Status
        self.status_label = QLabel("Ready")
        self.status_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.status_label)
    
    def select_file(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Image File",
            "",
            "Image Files (*.png *.jpg *.jpeg *.bmp *.tiff *.tif);;All Files (*.*)"
        )
        
        if file_path:
            self.current_file_path = file_path
            self.file_label.setText(os.path.basename(file_path))
            self.process_button.setEnabled(True)
            self.status_label.setText("File selected. Click 'Extract Text' to process.")
            
            # Show preview
            pixmap = QPixmap(file_path)
            if not pixmap.isNull():
                scaled_pixmap = pixmap.scaled(400, 300, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                self.preview_label.setPixmap(scaled_pixmap)
    
    def process_ocr(self):
        if not self.current_file_path:
            return
            
        try:
            self.status_label.setText("Processing...")
            
            # Load and preprocess image
            image = Image.open(self.current_file_path)
            
            # Enhance image for better OCR
            enhancer = ImageEnhance.Contrast(image)
            image = enhancer.enhance(2.0)
            enhancer = ImageEnhance.Sharpness(image)
            image = enhancer.enhance(2.0)
            
            # Perform OCR
            text = pytesseract.image_to_string(image)
            
            # Display results
            self.results_text.setText(text)
            self.status_label.setText(f"OCR completed. Extracted {len(text)} characters.")
            
        except Exception as e:
            self.status_label.setText(f"Error: {str(e)}")
            QMessageBox.critical(self, "OCR Error", f"Failed to process image: {str(e)}")


def main():
    # Check if Tesseract is available
    try:
        pytesseract.get_tesseract_version()
    except pytesseract.TesseractNotFoundError:
        print("Error: Tesseract OCR is not installed or not in system PATH.")
        print("Please install Tesseract OCR:")
        print("  macOS: brew install tesseract")
        print("  Windows: Download from https://github.com/UB-Mannheim/tesseract/wiki")
        print("  Linux: sudo apt-get install tesseract-ocr")
        return
    
    app = QApplication(sys.argv)
    window = SimpleOCRSystem()
    window.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()