import sys
import cv2
import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, 
                             QWidget, QLabel, QPushButton, QFileDialog, QSlider,
                             QGroupBox, QSpinBox, QComboBox, QProgressBar)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QThread
from PyQt5.QtGui import QImage, QPixmap, QFont
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


class DepthEstimationWorker(QThread):
    """Worker thread for depth estimation processing"""
    update_frame = pyqtSignal(np.ndarray, np.ndarray)
    update_progress = pyqtSignal(int)
    
    def __init__(self, video_source=0):
        super().__init__()
        self.video_source = video_source
        self.running = False
        self.cap = None
        
    def run(self):
        self.running = True
        self.cap = cv2.VideoCapture(self.video_source)
        
        while self.running and self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break
                
            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Simulate depth estimation (replace with actual DepthEstimationV2)
            depth_map = self.simulate_depth_estimation(frame)
            
            self.update_frame.emit(rgb_frame, depth_map)
            
        if self.cap:
            self.cap.release()
            
    def simulate_depth_estimation(self, frame):
        """Simulate depth estimation - replace with actual DepthEstimationV2 system"""
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Simulate depth using edge detection and distance transform
        edges = cv2.Canny(gray, 50, 150)
        
        # Create a simulated depth map
        height, width = frame.shape[:2]
        depth_map = np.zeros((height, width), dtype=np.float32)
        
        # Create center-based depth (objects in center are closer)
        y_coords, x_coords = np.ogrid[:height, :width]
        center_x, center_y = width // 2, height // 2
        
        # Distance from center (inverted for depth effect)
        distance_from_center = np.sqrt((x_coords - center_x)**2 + (y_coords - center_y)**2)
        max_distance = np.sqrt(center_x**2 + center_y**2)
        
        # Base depth based on distance from center
        depth_map = 1.0 - (distance_from_center / max_distance)
        
        # Enhance depth with edges
        depth_map[edges > 0] *= 1.5
        depth_map = np.clip(depth_map, 0, 1)
        
        # Apply colormap for visualization
        depth_colored = cv2.applyColorMap((depth_map * 255).astype(np.uint8), cv2.COLORMAP_JET)
        
        return depth_colored
    
    def stop(self):
        self.running = False
        self.wait()


class DepthEstimationInterface(QMainWindow):
    def __init__(self):
        super().__init__()
        self.worker = None
        self.is_processing = False
        self.init_ui()
        
    def init_ui(self):
        self.setWindowTitle("DepthEstimationV2 System Interface")
        self.setGeometry(100, 100, 1400, 800)
        
        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Main layout
        main_layout = QHBoxLayout()
        central_widget.setLayout(main_layout)
        
        # Left panel for controls
        left_panel = self.create_control_panel()
        main_layout.addWidget(left_panel, 1)
        
        # Right panel for display
        right_panel = self.create_display_panel()
        main_layout.addWidget(right_panel, 2)
        
    def create_control_panel(self):
        panel = QWidget()
        panel.setMaximumWidth(300)
        layout = QVBoxLayout()
        
        # Title
        title = QLabel("DepthEstimationV2")
        title.setFont(QFont("Arial", 16, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)
        
        # Input source group
        input_group = QGroupBox("Input Source")
        input_layout = QVBoxLayout()
        
        self.source_combo = QComboBox()
        self.source_combo.addItems(["Webcam", "Image File", "Video File"])
        input_layout.addWidget(QLabel("Source Type:"))
        input_layout.addWidget(self.source_combo)
        
        self.file_button = QPushButton("Select File")
        self.file_button.clicked.connect(self.select_file)
        input_layout.addWidget(self.file_button)
        
        input_group.setLayout(input_layout)
        layout.addWidget(input_group)
        
        # Processing controls group
        control_group = QGroupBox("Processing Controls")
        control_layout = QVBoxLayout()
        
        self.start_button = QPushButton("Start Processing")
        self.start_button.clicked.connect(self.toggle_processing)
        control_layout.addWidget(self.start_button)
        
        self.stop_button = QPushButton("Stop Processing")
        self.stop_button.clicked.connect(self.stop_processing)
        self.stop_button.setEnabled(False)
        control_layout.addWidget(self.stop_button)
        
        # Depth parameters
        control_layout.addWidget(QLabel("Depth Range:"))
        self.depth_slider = QSlider(Qt.Horizontal)
        self.depth_slider.setRange(1, 10)
        self.depth_slider.setValue(5)
        control_layout.addWidget(self.depth_slider)
        
        control_layout.addWidget(QLabel("Smoothness:"))
        self.smooth_slider = QSlider(Qt.Horizontal)
        self.smooth_slider.setRange(1, 10)
        self.smooth_slider.setValue(3)
        control_layout.addWidget(self.smooth_slider)
        
        control_group.setLayout(control_layout)
        layout.addWidget(control_group)
        
        # Visualization settings
        viz_group = QGroupBox("Visualization")
        viz_layout = QVBoxLayout()
        
        self.colormap_combo = QComboBox()
        self.colormap_combo.addItems(["JET", "VIRIDIS", "PLASMA", "INFERNO", "MAGMA"])
        viz_layout.addWidget(QLabel("Color Map:"))
        viz_layout.addWidget(self.colormap_combo)
        
        self.overlay_checkbox = QPushButton("Toggle Overlay")
        self.overlay_checkbox.setCheckable(True)
        self.overlay_checkbox.clicked.connect(self.toggle_overlay)
        viz_layout.addWidget(self.overlay_checkbox)
        
        viz_group.setLayout(viz_layout)
        layout.addWidget(viz_group)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)
        
        # Status
        self.status_label = QLabel("Ready")
        self.status_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.status_label)
        
        layout.addStretch()
        panel.setLayout(layout)
        return panel
    
    def create_display_panel(self):
        panel = QWidget()
        layout = QVBoxLayout()
        
        # Display area for images
        display_layout = QHBoxLayout()
        
        # Original image
        self.original_label = QLabel()
        self.original_label.setAlignment(Qt.AlignCenter)
        self.original_label.setMinimumSize(400, 300)
        self.original_label.setStyleSheet("border: 1px solid gray;")
        self.original_label.setText("Original Image")
        
        # Depth map
        self.depth_label = QLabel()
        self.depth_label.setAlignment(Qt.AlignCenter)
        self.depth_label.setMinimumSize(400, 300)
        self.depth_label.setStyleSheet("border: 1px solid gray;")
        self.depth_label.setText("Depth Map")
        
        display_layout.addWidget(self.original_label)
        display_layout.addWidget(self.depth_label)
        layout.addLayout(display_layout)
        
        # Depth histogram
        self.histogram_canvas = FigureCanvas(Figure(figsize=(8, 2)))
        layout.addWidget(self.histogram_canvas)
        
        # Statistics
        stats_group = QGroupBox("Depth Statistics")
        stats_layout = QHBoxLayout()
        
        self.min_depth_label = QLabel("Min: --")
        self.max_depth_label = QLabel("Max: --")
        self.avg_depth_label = QLabel("Avg: --")
        
        stats_layout.addWidget(self.min_depth_label)
        stats_layout.addWidget(self.avg_depth_label)
        stats_layout.addWidget(self.max_depth_label)
        
        stats_group.setLayout(stats_layout)
        layout.addWidget(stats_group)
        
        panel.setLayout(layout)
        return panel
    
    def select_file(self):
        source_type = self.source_combo.currentText()
        if source_type == "Image File":
            file_path, _ = QFileDialog.getOpenFileName(
                self, "Select Image", "", "Image Files (*.png *.jpg *.jpeg *.bmp)")
            if file_path:
                self.process_single_image(file_path)
        elif source_type == "Video File":
            file_path, _ = QFileDialog.getOpenFileName(
                self, "Select Video", "", "Video Files (*.mp4 *.avi *.mov)")
            if file_path:
                self.start_video_processing(file_path)
    
    def process_single_image(self, image_path):
        """Process a single image and display results"""
        image = cv2.imread(image_path)
        if image is not None:
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            depth_map = self.simulate_depth_estimation(image)
            
            self.display_images(rgb_image, depth_map)
            self.update_depth_statistics(depth_map)
            self.update_histogram(depth_map)
    
    def simulate_depth_estimation(self, image):
        """Simulate depth estimation for single image"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        height, width = image.shape[:2]
        
        # Create simulated depth map
        depth_map = np.zeros((height, width), dtype=np.float32)
        y_coords, x_coords = np.ogrid[:height, :width]
        center_x, center_y = width // 2, height // 2
        
        distance_from_center = np.sqrt((x_coords - center_x)**2 + (y_coords - center_y)**2)
        max_distance = np.sqrt(center_x**2 + center_y**2)
        
        depth_map = 1.0 - (distance_from_center / max_distance)
        
        # Add some object-based depth variations
        edges = cv2.Canny(gray, 50, 150)
        depth_map[edges > 0] *= 1.3
        depth_map = np.clip(depth_map, 0, 1)
        
        depth_colored = cv2.applyColorMap((depth_map * 255).astype(np.uint8), cv2.COLORMAP_JET)
        return depth_colored
    
    def toggle_processing(self):
        if not self.is_processing:
            self.start_processing()
        else:
            self.stop_processing()
    
    def start_processing(self):
        source_type = self.source_combo.currentText()
        
        if source_type == "Webcam":
            self.worker = DepthEstimationWorker(0)
        else:
            self.status_label.setText("Please select a file first")
            return
            
        self.worker.update_frame.connect(self.update_display)
        self.worker.update_progress.connect(self.progress_bar.setValue)
        
        self.worker.start()
        self.is_processing = True
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.progress_bar.setVisible(True)
        self.status_label.setText("Processing...")
    
    def start_video_processing(self, video_path):
        self.worker = DepthEstimationWorker(video_path)
        self.worker.update_frame.connect(self.update_display)
        self.worker.start()
        self.is_processing = True
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.status_label.setText("Processing video...")
    
    def stop_processing(self):
        if self.worker and self.worker.isRunning():
            self.worker.stop()
            self.worker.wait()
            
        self.is_processing = False
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.progress_bar.setVisible(False)
        self.status_label.setText("Stopped")
    
    def update_display(self, original_frame, depth_map):
        """Update the display with new frames"""
        self.display_images(original_frame, depth_map)
        self.update_depth_statistics(depth_map)
        self.update_histogram(depth_map)
    
    def display_images(self, original, depth):
        """Display original and depth images"""
        # Convert numpy arrays to QPixmap
        original_qimage = self.numpy_to_qimage(original)
        depth_qimage = self.numpy_to_qimage(depth)
        
        # Scale images to fit labels
        original_pixmap = QPixmap.fromImage(original_qimage).scaled(
            self.original_label.width(), 
            self.original_label.height(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )
        
        depth_pixmap = QPixmap.fromImage(depth_qimage).scaled(
            self.depth_label.width(), 
            self.depth_label.height(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )
        
        self.original_label.setPixmap(original_pixmap)
        self.depth_label.setPixmap(depth_pixmap)
    
    def numpy_to_qimage(self, array):
        """Convert numpy array to QImage"""
        height, width, channel = array.shape
        bytes_per_line = 3 * width
        return QImage(array.data, width, height, bytes_per_line, QImage.Format_RGB888)
    
    def update_depth_statistics(self, depth_map):
        """Update depth statistics display"""
        # Convert color depth map to grayscale for statistics
        depth_gray = cv2.cvtColor(depth_map, cv2.COLOR_RGB2GRAY)
        depth_values = depth_gray.astype(np.float32) / 255.0
        
        min_depth = np.min(depth_values)
        max_depth = np.max(depth_values)
        avg_depth = np.mean(depth_values)
        
        self.min_depth_label.setText(f"Min: {min_depth:.3f}")
        self.max_depth_label.setText(f"Max: {max_depth:.3f}")
        self.avg_depth_label.setText(f"Avg: {avg_depth:.3f}")
    
    def update_histogram(self, depth_map):
        """Update depth histogram"""
        depth_gray = cv2.cvtColor(depth_map, cv2.COLOR_RGB2GRAY)
        depth_values = depth_gray.astype(np.float32) / 255.0
        
        self.histogram_canvas.figure.clear()
        ax = self.histogram_canvas.figure.add_subplot(111)
        ax.hist(depth_values.flatten(), bins=50, alpha=0.7, color='blue')
        ax.set_xlabel('Depth Value')
        ax.set_ylabel('Frequency')
        ax.set_title('Depth Distribution')
        ax.grid(True, alpha=0.3)
        self.histogram_canvas.draw()
    
    def toggle_overlay(self):
        """Toggle overlay mode (placeholder)"""
        if self.overlay_checkbox.isChecked():
            self.status_label.setText("Overlay mode enabled")
        else:
            self.status_label.setText("Overlay mode disabled")
    
    def closeEvent(self, event):
        """Ensure proper cleanup when closing"""
        self.stop_processing()
        event.accept()


def main():
    app = QApplication(sys.argv)
    
    # Set application style
    app.setStyle('Fusion')
    
    window = DepthEstimationInterface()
    window.show()
    
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()