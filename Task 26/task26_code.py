# Access the camera with PyQt5. Assign IDs to objects in the camera view. Develop an effective control system with "Start Camera" and "Stop Camera" buttons in this interface.

import sys
import cv2
import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, 
                             QWidget, QLabel, QPushButton, QFileDialog, QSlider,
                             QGroupBox, QSpinBox, QComboBox, QProgressBar, QMessageBox,
                             QListWidget, QListWidgetItem, QCheckBox, QDoubleSpinBox)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QThread, QMutex, pyqtSlot
from PyQt5.QtGui import QImage, QPixmap, QFont, QColor
import random


class CameraProcessor(QThread):
    """Camera processing thread with object detection and tracking"""
    frame_ready = pyqtSignal(np.ndarray)
    objects_updated = pyqtSignal(list)
    camera_error = pyqtSignal(str)
    
    def __init__(self, camera_id=0):
        super().__init__()
        self.camera_id = camera_id
        self.running = False
        self.cap = None
        self.mutex = QMutex()
        self.objects = []
        self.next_object_id = 1
        self.detection_interval = 5  # frames between detections
        self.frame_count = 0
        
        # Object tracking parameters
        self.tracking_threshold = 50  # pixel distance for tracking
        self.min_object_area = 500    # minimum area for object detection
        
    def run(self):
        self.running = True
        try:
            self.cap = cv2.VideoCapture(self.camera_id)
            if not self.cap.isOpened():
                self.camera_error.emit(f"Could not open camera {self.camera_id}")
                return
                
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.cap.set(cv2.CAP_PROP_FPS, 30)
            
            while self.running:
                ret, frame = self.cap.read()
                if not ret:
                    self.camera_error.emit("Failed to capture frame")
                    break
                
                # Process frame for object detection and tracking
                processed_frame = self.process_frame(frame)
                
                # Emit the processed frame
                self.frame_ready.emit(processed_frame)
                
        except Exception as e:
            self.camera_error.emit(f"Camera error: {str(e)}")
        finally:
            if self.cap:
                self.cap.release()
    
    def process_frame(self, frame):
        """Process frame for object detection and tracking"""
        self.frame_count += 1
        
        # Perform object detection periodically
        if self.frame_count % self.detection_interval == 0:
            new_objects = self.detect_objects(frame)
            self.update_object_tracking(new_objects)
        
        # Draw objects on frame
        frame_with_objects = self.draw_objects(frame)
        
        return frame_with_objects
    
    def detect_objects(self, frame):
        """Detect objects in the frame using contour detection"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (7, 7), 0)
        
        # Adaptive thresholding for better object detection
        thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                      cv2.THRESH_BINARY_INV, 11, 2)
        
        # Morphological operations to clean up the image
        kernel = np.ones((5, 5), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        detected_objects = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > self.min_object_area:
                # Get bounding box and center
                x, y, w, h = cv2.boundingRect(contour)
                center_x = x + w // 2
                center_y = y + h // 2
                
                detected_objects.append({
                    'center': (center_x, center_y),
                    'bbox': (x, y, w, h),
                    'area': area
                })
        
        return detected_objects
    
    def update_object_tracking(self, new_objects):
        """Update object tracking with new detections"""
        self.mutex.lock()
        
        # If no existing objects, assign IDs to all new objects
        if not self.objects:
            for obj in new_objects:
                obj['id'] = self.next_object_id
                obj['color'] = self.generate_color(obj['id'])
                obj['tracking_points'] = [obj['center']]
                self.objects.append(obj)
                self.next_object_id += 1
        else:
            # Match new objects with existing ones
            matched_indices = set()
            
            for existing_obj in self.objects:
                best_match_idx = -1
                min_distance = float('inf')
                
                for i, new_obj in enumerate(new_objects):
                    if i in matched_indices:
                        continue
                    
                    # Calculate distance between centers
                    dist = np.sqrt((existing_obj['center'][0] - new_obj['center'][0])**2 + 
                                 (existing_obj['center'][1] - new_obj['center'][1])**2)
                    
                    if dist < self.tracking_threshold and dist < min_distance:
                        min_distance = dist
                        best_match_idx = i
                
                if best_match_idx != -1:
                    # Update existing object
                    matched_obj = new_objects[best_match_idx]
                    existing_obj['center'] = matched_obj['center']
                    existing_obj['bbox'] = matched_obj['bbox']
                    existing_obj['area'] = matched_obj['area']
                    existing_obj['tracking_points'].append(matched_obj['center'])
                    
                    # Keep only last 20 tracking points
                    if len(existing_obj['tracking_points']) > 20:
                        existing_obj['tracking_points'] = existing_obj['tracking_points'][-20:]
                    
                    matched_indices.add(best_match_idx)
            
            # Add new objects for unmatched detections
            for i, new_obj in enumerate(new_objects):
                if i not in matched_indices:
                    new_obj['id'] = self.next_object_id
                    new_obj['color'] = self.generate_color(self.next_object_id)
                    new_obj['tracking_points'] = [new_obj['center']]
                    self.objects.append(new_obj)
                    self.next_object_id += 1
        
        # Remove objects that haven't been updated (lost tracking)
        self.objects = [obj for obj in self.objects if any(
            obj['center'] == new_obj['center'] for new_obj in new_objects
        ) or len(new_objects) == 0]
        
        self.mutex.unlock()
        self.objects_updated.emit(self.objects.copy())
    
    def draw_objects(self, frame):
        """Draw objects and their IDs on the frame"""
        self.mutex.lock()
        objects_copy = self.objects.copy()
        self.mutex.unlock()
        
        for obj in objects_copy:
            color = obj['color']
            x, y, w, h = obj['bbox']
            center_x, center_y = obj['center']
            
            # Draw bounding box
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            
            # Draw object ID
            cv2.putText(frame, f"ID: {obj['id']}", (x, y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            # Draw center point
            cv2.circle(frame, (center_x, center_y), 5, color, -1)
            
            # Draw tracking path
            points = obj.get('tracking_points', [])
            for i in range(1, len(points)):
                cv2.line(frame, points[i-1], points[i], color, 2)
        
        return frame
    
    def generate_color(self, object_id):
        """Generate a consistent color for each object ID"""
        random.seed(object_id)
        return (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
    
    def stop_processing(self):
        """Stop the camera processing"""
        self.running = False
        self.wait()


class CameraControlInterface(QMainWindow):
    def __init__(self):
        super().__init__()
        self.camera_processor = None
        self.is_camera_running = False
        self.current_objects = []
        self.init_ui()
        
    def init_ui(self):
        self.setWindowTitle("Advanced Camera Control System with Object Tracking")
        self.setGeometry(100, 100, 1400, 900)
        
        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Main layout
        main_layout = QHBoxLayout()
        central_widget.setLayout(main_layout)
        
        # Left panel for controls
        left_panel = self.create_control_panel()
        main_layout.addWidget(left_panel, 1)
        
        # Right panel for display and object list
        right_panel = self.create_display_panel()
        main_layout.addWidget(right_panel, 2)
        
    def create_control_panel(self):
        panel = QWidget()
        panel.setMaximumWidth(400)
        layout = QVBoxLayout()
        
        # Title
        title = QLabel("Camera Control System")
        title.setFont(QFont("Arial", 16, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)
        
        # Camera Configuration Group
        config_group = QGroupBox("Camera Configuration")
        config_layout = QVBoxLayout()
        
        # Camera selection
        config_layout.addWidget(QLabel("Camera ID:"))
        self.camera_id_spinbox = QSpinBox()
        self.camera_id_spinbox.setRange(0, 10)
        self.camera_id_spinbox.setValue(0)
        config_layout.addWidget(self.camera_id_spinbox)
        
        # Resolution selection
        config_layout.addWidget(QLabel("Resolution:"))
        self.resolution_combo = QComboBox()
        self.resolution_combo.addItems(["640x480", "800x600", "1024x768", "1280x720"])
        self.resolution_combo.setCurrentText("640x480")
        config_layout.addWidget(self.resolution_combo)
        
        config_group.setLayout(config_layout)
        layout.addWidget(config_group)
        
        # Camera Control Group
        control_group = QGroupBox("Camera Control")
        control_layout = QVBoxLayout()
        
        # Start/Stop buttons
        button_layout = QHBoxLayout()
        self.start_button = QPushButton("Start Camera")
        self.start_button.clicked.connect(self.start_camera)
        self.start_button.setStyleSheet("QPushButton { background-color: #4CAF50; color: white; }")
        
        self.stop_button = QPushButton("Stop Camera")
        self.stop_button.clicked.connect(self.stop_camera)
        self.stop_button.setEnabled(False)
        self.stop_button.setStyleSheet("QPushButton { background-color: #f44336; color: white; }")
        
        button_layout.addWidget(self.start_button)
        button_layout.addWidget(self.stop_button)
        control_layout.addLayout(button_layout)
        
        # Camera status
        self.camera_status_label = QLabel("Camera: Stopped")
        self.camera_status_label.setAlignment(Qt.AlignCenter)
        self.camera_status_label.setStyleSheet("QLabel { background-color: #ffeb3b; padding: 5px; }")
        control_layout.addWidget(self.camera_status_label)
        
        control_group.setLayout(control_layout)
        layout.addWidget(control_group)
        
        # Object Detection Settings
        detection_group = QGroupBox("Object Detection Settings")
        detection_layout = QVBoxLayout()
        
        # Detection sensitivity
        detection_layout.addWidget(QLabel("Detection Sensitivity:"))
        self.sensitivity_slider = QSlider(Qt.Horizontal)
        self.sensitivity_slider.setRange(1, 10)
        self.sensitivity_slider.setValue(5)
        self.sensitivity_slider.valueChanged.connect(self.update_detection_params)
        detection_layout.addWidget(self.sensitivity_slider)
        
        # Minimum object size
        detection_layout.addWidget(QLabel("Minimum Object Size:"))
        self.min_size_spinbox = QSpinBox()
        self.min_size_spinbox.setRange(100, 5000)
        self.min_size_spinbox.setValue(500)
        self.min_size_spinbox.valueChanged.connect(self.update_detection_params)
        detection_layout.addWidget(self.min_size_spinbox)
        
        # Tracking threshold
        detection_layout.addWidget(QLabel("Tracking Threshold:"))
        self.tracking_threshold_spinbox = QSpinBox()
        self.tracking_threshold_spinbox.setRange(10, 200)
        self.tracking_threshold_spinbox.setValue(50)
        self.tracking_threshold_spinbox.valueChanged.connect(self.update_detection_params)
        detection_layout.addWidget(self.tracking_threshold_spinbox)
        
        detection_group.setLayout(detection_layout)
        layout.addWidget(detection_group)
        
        # Display Options
        display_group = QGroupBox("Display Options")
        display_layout = QVBoxLayout()
        
        self.show_bbox_checkbox = QCheckBox("Show Bounding Boxes")
        self.show_bbox_checkbox.setChecked(True)
        display_layout.addWidget(self.show_bbox_checkbox)
        
        self.show_tracking_checkbox = QCheckBox("Show Tracking Paths")
        self.show_tracking_checkbox.setChecked(True)
        display_layout.addWidget(self.show_tracking_checkbox)
        
        self.show_ids_checkbox = QCheckBox("Show Object IDs")
        self.show_ids_checkbox.setChecked(True)
        display_layout.addWidget(self.show_ids_checkbox)
        
        display_group.setLayout(display_layout)
        layout.addWidget(display_group)
        
        # Statistics Group
        stats_group = QGroupBox("System Statistics")
        stats_layout = QVBoxLayout()
        
        self.fps_label = QLabel("FPS: --")
        self.objects_count_label = QLabel("Objects Detected: 0")
        self.frame_count_label = QLabel("Frames Processed: 0")
        
        stats_layout.addWidget(self.fps_label)
        stats_layout.addWidget(self.objects_count_label)
        stats_layout.addWidget(self.frame_count_label)
        
        stats_group.setLayout(stats_layout)
        layout.addWidget(stats_group)
        
        layout.addStretch()
        panel.setLayout(layout)
        return panel
    
    def create_display_panel(self):
        panel = QWidget()
        layout = QVBoxLayout()
        
        # Camera feed display
        self.camera_label = QLabel()
        self.camera_label.setAlignment(Qt.AlignCenter)
        self.camera_label.setMinimumSize(640, 480)
        self.camera_label.setStyleSheet("border: 2px solid gray; background-color: black;")
        self.camera_label.setText("Camera Feed\nClick 'Start Camera' to begin")
        layout.addWidget(self.camera_label)
        
        # Object list
        objects_group = QGroupBox("Detected Objects")
        objects_layout = QVBoxLayout()
        
        self.objects_list = QListWidget()
        self.objects_list.itemClicked.connect(self.on_object_selected)
        objects_layout.addWidget(self.objects_list)
        
        # Object actions
        object_actions_layout = QHBoxLayout()
        self.clear_objects_button = QPushButton("Clear Objects")
        self.clear_objects_button.clicked.connect(self.clear_objects)
        self.export_objects_button = QPushButton("Export Objects")
        self.export_objects_button.clicked.connect(self.export_objects)
        
        object_actions_layout.addWidget(self.clear_objects_button)
        object_actions_layout.addWidget(self.export_objects_button)
        objects_layout.addLayout(object_actions_layout)
        
        objects_group.setLayout(objects_layout)
        layout.addWidget(objects_group)
        
        panel.setLayout(layout)
        return panel
    
    def start_camera(self):
        """Start the camera processing"""
        if self.is_camera_running:
            return
        
        camera_id = self.camera_id_spinbox.value()
        
        try:
            self.camera_processor = CameraProcessor(camera_id)
            self.camera_processor.frame_ready.connect(self.update_camera_feed)
            self.camera_processor.objects_updated.connect(self.update_objects_list)
            self.camera_processor.camera_error.connect(self.handle_camera_error)
            
            # Update detection parameters
            self.update_detection_params()
            
            self.camera_processor.start()
            self.is_camera_running = True
            
            # Update UI
            self.start_button.setEnabled(False)
            self.stop_button.setEnabled(True)
            self.camera_status_label.setText("Camera: Running")
            self.camera_status_label.setStyleSheet("QLabel { background-color: #4CAF50; color: white; padding: 5px; }")
            self.camera_label.setText("Initializing camera...")
            
        except Exception as e:
            QMessageBox.critical(self, "Camera Error", f"Failed to start camera: {str(e)}")
    
    def stop_camera(self):
        """Stop the camera processing"""
        if not self.is_camera_running:
            return
        
        if self.camera_processor:
            self.camera_processor.stop_processing()
            self.camera_processor.wait()
            self.camera_processor = None
        
        self.is_camera_running = False
        
        # Update UI
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.camera_status_label.setText("Camera: Stopped")
        self.camera_status_label.setStyleSheet("QLabel { background-color: #ffeb3b; padding: 5px; }")
        self.camera_label.setText("Camera Feed\nClick 'Start Camera' to begin")
        
        # Clear objects list
        self.objects_list.clear()
        self.current_objects = []
        self.update_objects_count()
    
    def update_camera_feed(self, frame):
        """Update the camera feed display"""
        try:
            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Convert to QImage
            h, w, ch = rgb_frame.shape
            bytes_per_line = ch * w
            q_img = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
            
            # Scale image to fit label while maintaining aspect ratio
            pixmap = QPixmap.fromImage(q_img)
            scaled_pixmap = pixmap.scaled(self.camera_label.width(), 
                                        self.camera_label.height(),
                                        Qt.KeepAspectRatio, 
                                        Qt.SmoothTransformation)
            
            self.camera_label.setPixmap(scaled_pixmap)
            
        except Exception as e:
            print(f"Error updating camera feed: {e}")
    
    def update_objects_list(self, objects):
        """Update the objects list with current detected objects"""
        self.current_objects = objects
        self.objects_list.clear()
        
        for obj in objects:
            x, y, w, h = obj['bbox']
            item_text = f"ID: {obj['id']} | Position: ({x}, {y}) | Size: {w}x{h} | Area: {obj['area']:.0f}"
            item = QListWidgetItem(item_text)
            
            # Set item color based on object color
            color = obj['color']
            item.setBackground(QColor(color[2], color[1], color[0]))  # RGB to BGR for QColor
            item.setForeground(QColor(255, 255, 255))  # White text
            
            self.objects_list.addItem(item)
        
        self.update_objects_count()
    
    def update_objects_count(self):
        """Update the objects count display"""
        count = len(self.current_objects)
        self.objects_count_label.setText(f"Objects Detected: {count}")
    
    def update_detection_params(self):
        """Update object detection parameters"""
        if self.camera_processor:
            sensitivity = self.sensitivity_slider.value()
            min_size = self.min_size_spinbox.value()
            tracking_threshold = self.tracking_threshold_spinbox.value()
            
            # Update parameters based on sensitivity
            self.camera_processor.min_object_area = min_size
            self.camera_processor.tracking_threshold = tracking_threshold
            self.camera_processor.detection_interval = max(1, 10 - sensitivity + 1)
    
    def on_object_selected(self, item):
        """Handle object selection from the list"""
        # You can implement additional actions when an object is selected
        pass
    
    def clear_objects(self):
        """Clear the objects list"""
        if self.camera_processor:
            self.camera_processor.objects.clear()
            self.camera_processor.next_object_id = 1
        self.objects_list.clear()
        self.current_objects = []
        self.update_objects_count()
    
    def export_objects(self):
        """Export object data to file"""
        if not self.current_objects:
            QMessageBox.information(self, "Export", "No objects to export")
            return
        
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Export Object Data", "", "Text Files (*.txt);;CSV Files (*.csv)")
        
        if file_path:
            try:
                with open(file_path, 'w') as f:
                    f.write("Object ID,Position X,Position Y,Width,Height,Area\n")
                    for obj in self.current_objects:
                        x, y, w, h = obj['bbox']
                        f.write(f"{obj['id']},{x},{y},{w},{h},{obj['area']:.2f}\n")
                
                QMessageBox.information(self, "Export Successful", 
                                      f"Object data exported to:\n{file_path}")
            except Exception as e:
                QMessageBox.critical(self, "Export Error", f"Failed to export: {str(e)}")
    
    def handle_camera_error(self, error_message):
        """Handle camera errors"""
        QMessageBox.critical(self, "Camera Error", error_message)
        self.stop_camera()
    
    def closeEvent(self, event):
        """Ensure proper cleanup when closing"""
        self.stop_camera()
        event.accept()


def main():
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    
    window = CameraControlInterface()
    window.show()
    
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
    