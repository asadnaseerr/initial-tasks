# Add a button to the interface you created for DepthEstimationV2 and output a 3D model from the image using the data in the depth map.

import sys
import cv2
import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, 
                             QWidget, QLabel, QPushButton, QFileDialog, QSlider,
                             QGroupBox, QSpinBox, QComboBox, QProgressBar, QMessageBox)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QThread
from PyQt5.QtGui import QImage, QPixmap, QFont
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D
import open3d as o3d
import tempfile
import os
from stl import mesh
import trimesh


class ThreeDModelGenerator(QThread):
    """Thread for generating 3D models from depth maps"""
    model_generated = pyqtSignal(str)  # Emits file path when done
    progress_updated = pyqtSignal(int)
    
    def __init__(self, original_image, depth_map, method='point_cloud'):
        super().__init__()
        self.original_image = original_image
        self.depth_map = depth_map
        self.method = method
        
    def run(self):
        try:
            self.progress_updated.emit(10)
            
            if self.method == 'point_cloud':
                file_path = self.generate_point_cloud()
            elif self.method == 'mesh':
                file_path = self.generate_mesh()
            elif self.method == 'height_map':
                file_path = self.generate_height_map_mesh()
            else:
                file_path = self.generate_point_cloud()
                
            self.progress_updated.emit(100)
            self.model_generated.emit(file_path)
            
        except Exception as e:
            print(f"Error generating 3D model: {e}")
            self.model_generated.emit("")
    
    def generate_point_cloud(self):
        """Generate a 3D point cloud from depth map"""
        self.progress_updated.emit(30)
        
        # Convert depth map to grayscale and normalize
        depth_gray = cv2.cvtColor(self.depth_map, cv2.COLOR_RGB2GRAY)
        depth_normalized = depth_gray.astype(np.float32) / 255.0
        
        height, width = depth_normalized.shape
        
        # Create point cloud
        points = []
        colors = []
        
        # Downsample for performance
        step = 2
        for y in range(0, height, step):
            for x in range(0, width, step):
                z = depth_normalized[y, x] * 10.0  # Scale depth
                points.append([x - width/2, y - height/2, z * 100])
                
                # Get color from original image
                if y < self.original_image.shape[0] and x < self.original_image.shape[1]:
                    color = self.original_image[y, x] / 255.0
                    colors.append(color)
                else:
                    colors.append([1.0, 1.0, 1.0])
        
        self.progress_updated.emit(60)
        
        # Create Open3D point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(np.array(points))
        pcd.colors = o3d.utility.Vector3dVector(np.array(colors))
        
        # Save point cloud
        temp_dir = tempfile.gettempdir()
        file_path = os.path.join(temp_dir, f"point_cloud_{np.random.randint(10000)}.ply")
        o3d.io.write_point_cloud(file_path, pcd)
        
        self.progress_updated.emit(90)
        return file_path
    
    def generate_height_map_mesh(self):
        """Generate a 3D mesh using height map approach"""
        self.progress_updated.emit(30)
        
        # Convert depth map to grayscale and normalize
        depth_gray = cv2.cvtColor(self.depth_map, cv2.COLOR_RGB2GRAY)
        depth_normalized = depth_gray.astype(np.float32) / 255.0
        
        height, width = depth_normalized.shape
        
        # Downsample for performance
        scale = 0.5
        new_width = int(width * scale)
        new_height = int(height * scale)
        depth_resized = cv2.resize(depth_normalized, (new_width, new_height))
        
        # Create vertices
        vertices = []
        for y in range(new_height):
            for x in range(new_width):
                z = depth_resized[y, x] * 20.0  # Height scale
                vertices.append([x - new_width/2, y - new_height/2, z * 50])
        
        self.progress_updated.emit(60)
        
        # Create faces
        faces = []
        for y in range(new_height - 1):
            for x in range(new_width - 1):
                v1 = y * new_width + x
                v2 = y * new_width + (x + 1)
                v3 = (y + 1) * new_width + x
                v4 = (y + 1) * new_width + (x + 1)
                
                # Two triangles per quad
                faces.append([v1, v2, v3])
                faces.append([v2, v4, v3])
        
        # Create mesh
        mesh_data = mesh.Mesh(np.zeros(len(faces), dtype=mesh.Mesh.dtype))
        for i, face in enumerate(faces):
            for j in range(3):
                mesh_data.vectors[i][j] = vertices[face[j]]
        
        self.progress_updated.emit(80)
        
        # Save STL file
        temp_dir = tempfile.gettempdir()
        file_path = os.path.join(temp_dir, f"mesh_{np.random.randint(10000)}.stl")
        mesh_data.save(file_path)
        
        return file_path
    
    def generate_mesh(self):
        """Generate a 3D mesh using Poisson reconstruction"""
        self.progress_updated.emit(20)
        
        # First create point cloud
        depth_gray = cv2.cvtColor(self.depth_map, cv2.COLOR_RGB2GRAY)
        depth_normalized = depth_gray.astype(np.float32) / 255.0
        
        height, width = depth_normalized.shape
        
        points = []
        colors = []
        
        step = 2
        for y in range(0, height, step):
            for x in range(0, width, step):
                z = depth_normalized[y, x] * 15.0
                points.append([x - width/2, y - height/2, z * 80])
                
                if y < self.original_image.shape[0] and x < self.original_image.shape[1]:
                    color = self.original_image[y, x] / 255.0
                    colors.append(color)
        
        self.progress_updated.emit(50)
        
        # Create point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(np.array(points))
        pcd.colors = o3d.utility.Vector3dVector(np.array(colors))
        
        # Estimate normals
        pcd.estimate_normals()
        
        self.progress_updated.emit(70)
        
        # Poisson reconstruction
        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            pcd, depth=8)
        
        self.progress_updated.emit(90)
        
        # Save mesh
        temp_dir = tempfile.gettempdir()
        file_path = os.path.join(temp_dir, f"poisson_mesh_{np.random.randint(10000)}.ply")
        o3d.io.write_triangle_mesh(file_path, mesh)
        
        return file_path


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
            
            # Simulate depth estimation
            depth_map = self.simulate_depth_estimation(frame)
            
            self.update_frame.emit(rgb_frame, depth_map)
            
        if self.cap:
            self.cap.release()
            
    def simulate_depth_estimation(self, frame):
        """Simulate depth estimation - replace with actual DepthEstimationV2 system"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        height, width = frame.shape[:2]
        
        depth_map = np.zeros((height, width), dtype=np.float32)
        y_coords, x_coords = np.ogrid[:height, :width]
        center_x, center_y = width // 2, height // 2
        
        distance_from_center = np.sqrt((x_coords - center_x)**2 + (y_coords - center_y)**2)
        max_distance = np.sqrt(center_x**2 + center_y**2)
        
        depth_map = 1.0 - (distance_from_center / max_distance)
        
        edges = cv2.Canny(gray, 50, 150)
        depth_map[edges > 0] *= 1.5
        depth_map = np.clip(depth_map, 0, 1)
        
        depth_colored = cv2.applyColorMap((depth_map * 255).astype(np.uint8), cv2.COLORMAP_JET)
        return depth_colored
    
    def stop(self):
        self.running = False
        self.wait()


class DepthEstimationInterface(QMainWindow):
    def __init__(self):
        super().__init__()
        self.worker = None
        self.model_generator = None
        self.is_processing = False
        self.current_original = None
        self.current_depth = None
        self.init_ui()
        
    def init_ui(self):
        self.setWindowTitle("DepthEstimationV2 System Interface - 3D Model Generation")
        self.setGeometry(100, 100, 1600, 900)
        
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
        panel.setMaximumWidth(350)
        layout = QVBoxLayout()
        
        # Title
        title = QLabel("DepthEstimationV2 + 3D")
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
        
        # 3D Model Generation Group
        model_group = QGroupBox("3D Model Generation")
        model_layout = QVBoxLayout()
        
        self.model_method_combo = QComboBox()
        self.model_method_combo.addItems(["Point Cloud", "Mesh", "Height Map"])
        model_layout.addWidget(QLabel("3D Model Type:"))
        model_layout.addWidget(self.model_method_combo)
        
        self.generate_3d_button = QPushButton("Generate 3D Model")
        self.generate_3d_button.clicked.connect(self.generate_3d_model)
        self.generate_3d_button.setEnabled(False)
        model_layout.addWidget(self.generate_3d_button)
        
        self.export_3d_button = QPushButton("Export 3D Model...")
        self.export_3d_button.clicked.connect(self.export_3d_model)
        self.export_3d_button.setEnabled(False)
        model_layout.addWidget(self.export_3d_button)
        
        self.preview_3d_button = QPushButton("Preview 3D Model")
        self.preview_3d_button.clicked.connect(self.preview_3d_model)
        self.preview_3d_button.setEnabled(False)
        model_layout.addWidget(self.preview_3d_button)
        
        model_group.setLayout(model_layout)
        layout.addWidget(model_group)
        
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
        
        # Progress bars
        self.processing_progress = QProgressBar()
        self.processing_progress.setVisible(False)
        layout.addWidget(QLabel("Processing Progress:"))
        layout.addWidget(self.processing_progress)
        
        self.model_progress = QProgressBar()
        self.model_progress.setVisible(False)
        layout.addWidget(QLabel("3D Model Progress:"))
        layout.addWidget(self.model_progress)
        
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
        original_group = QGroupBox("Original Image")
        original_layout = QVBoxLayout()
        self.original_label = QLabel()
        self.original_label.setAlignment(Qt.AlignCenter)
        self.original_label.setMinimumSize(400, 300)
        self.original_label.setStyleSheet("border: 1px solid gray;")
        self.original_label.setText("Original Image")
        original_layout.addWidget(self.original_label)
        original_group.setLayout(original_layout)
        
        # Depth map
        depth_group = QGroupBox("Depth Map")
        depth_layout = QVBoxLayout()
        self.depth_label = QLabel()
        self.depth_label.setAlignment(Qt.AlignCenter)
        self.depth_label.setMinimumSize(400, 300)
        self.depth_label.setStyleSheet("border: 1px solid gray;")
        self.depth_label.setText("Depth Map")
        depth_layout.addWidget(self.depth_label)
        depth_group.setLayout(depth_layout)
        
        display_layout.addWidget(original_group)
        display_layout.addWidget(depth_group)
        layout.addLayout(display_layout)
        
        # 3D Preview
        self.preview_3d_canvas = FigureCanvas(Figure(figsize=(8, 4)))
        layout.addWidget(QLabel("3D Model Preview:"))
        layout.addWidget(self.preview_3d_canvas)
        
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
    
    def generate_3d_model(self):
        """Generate 3D model from current depth map"""
        if self.current_original is None or self.current_depth is None:
            QMessageBox.warning(self, "Warning", "No image data available. Please process an image first.")
            return
        
        method_map = {
            "Point Cloud": "point_cloud",
            "Mesh": "mesh", 
            "Height Map": "height_map"
        }
        
        method = method_map[self.model_method_combo.currentText()]
        
        self.model_progress.setVisible(True)
        self.model_progress.setValue(0)
        self.status_label.setText("Generating 3D model...")
        
        self.model_generator = ThreeDModelGenerator(
            self.current_original, 
            self.current_depth, 
            method
        )
        self.model_generator.model_generated.connect(self.on_model_generated)
        self.model_generator.progress_updated.connect(self.model_progress.setValue)
        self.model_generator.start()
    
    def on_model_generated(self, file_path):
        """Handle completed 3D model generation"""
        self.model_progress.setVisible(False)
        
        if file_path:
            self.current_model_path = file_path
            self.export_3d_button.setEnabled(True)
            self.preview_3d_button.setEnabled(True)
            self.status_label.setText(f"3D model generated: {os.path.basename(file_path)}")
            
            # Show preview
            self.preview_3d_model()
        else:
            QMessageBox.critical(self, "Error", "Failed to generate 3D model")
            self.status_label.setText("3D model generation failed")
    
    def preview_3d_model(self):
        """Show 3D model preview"""
        if not hasattr(self, 'current_model_path') or not self.current_model_path:
            return
        
        try:
            # Clear previous preview
            self.preview_3d_canvas.figure.clear()
            
            # Create 3D subplot
            ax = self.preview_3d_canvas.figure.add_subplot(111, projection='3d')
            
            # Load and plot the 3D model based on file type
            if self.current_model_path.endswith('.ply'):
                if 'point_cloud' in self.current_model_path:
                    # Point cloud visualization
                    pcd = o3d.io.read_point_cloud(self.current_model_path)
                    points = np.asarray(pcd.points)
                    colors = np.asarray(pcd.colors)
                    
                    ax.scatter(points[:, 0], points[:, 1], points[:, 2], 
                              c=colors, s=1, alpha=0.6)
                else:
                    # Mesh visualization
                    mesh = o3d.io.read_triangle_mesh(self.current_model_path)
                    vertices = np.asarray(mesh.vertices)
                    triangles = np.asarray(mesh.triangles)
                    
                    ax.plot_trisurf(vertices[:, 0], vertices[:, 1], vertices[:, 2],
                                   triangles=triangles, alpha=0.8, cmap='viridis')
            
            elif self.current_model_path.endswith('.stl'):
                # STL mesh visualization
                stl_mesh = mesh.Mesh.from_file(self.current_model_path)
                vertices = stl_mesh.vectors
                
                for triangle in vertices:
                    ax.plot_trisurf(triangle[:, 0], triangle[:, 1], triangle[:, 2],
                                   alpha=0.8, color='lightblue')
            
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            ax.set_title('3D Model Preview')
            
            self.preview_3d_canvas.draw()
            
        except Exception as e:
            QMessageBox.warning(self, "Preview Error", f"Could not generate preview: {str(e)}")
    
    def export_3d_model(self):
        """Export 3D model to user-selected location"""
        if not hasattr(self, 'current_model_path') or not self.current_model_path:
            return
        
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Export 3D Model", "", 
            "3D Files (*.ply *.stl *.obj);;Point Clouds (*.ply);;Meshes (*.stl *.obj)"
        )
        
        if file_path:
            try:
                import shutil
                shutil.copy2(self.current_model_path, file_path)
                QMessageBox.information(self, "Success", f"3D model exported to:\n{file_path}")
                self.status_label.setText(f"Model exported: {os.path.basename(file_path)}")
            except Exception as e:
                QMessageBox.critical(self, "Export Error", f"Failed to export model: {str(e)}")
    
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
            
            self.current_original = rgb_image
            self.current_depth = depth_map
            
            self.display_images(rgb_image, depth_map)
            self.update_depth_statistics(depth_map)
            self.update_histogram(depth_map)
            self.generate_3d_button.setEnabled(True)
    
    def simulate_depth_estimation(self, image):
        """Simulate depth estimation for single image"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        height, width = image.shape[:2]
        
        depth_map = np.zeros((height, width), dtype=np.float32)
        y_coords, x_coords = np.ogrid[:height, :width]
        center_x, center_y = width // 2, height // 2
        
        distance_from_center = np.sqrt((x_coords - center_x)**2 + (y_coords - center_y)**2)
        max_distance = np.sqrt(center_x**2 + center_y**2)
        
        depth_map = 1.0 - (distance_from_center / max_distance)
        
        edges = cv2.Canny(gray, 50, 150)
        depth_map[edges > 0] *= 1.3
        depth_map = np.clip(depth_map, 0, 1)
        
        depth_colored = cv2.applyColorMap((depth_map * 255).astype(np.uint8), cv2.COLORMAP_JET)
        return depth_colored
    
    def update_display(self, original_frame, depth_map):
        """Update the display with new frames"""
        self.current_original = original_frame
        self.current_depth = depth_map
        
        self.display_images(original_frame, depth_map)
        self.update_depth_statistics(depth_map)
        self.update_histogram(depth_map)
        self.generate_3d_button.setEnabled(True)
    
    # ... (rest of the methods remain the same as previous implementation)
    def display_images(self, original, depth):
        """Display original and depth images"""
        original_qimage = self.numpy_to_qimage(original)
        depth_qimage = self.numpy_to_qimage(depth)
        
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
        self.worker.update_progress.connect(self.processing_progress.setValue)
        
        self.worker.start()
        self.is_processing = True
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.processing_progress.setVisible(True)
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
        self.processing_progress.setVisible(False)
        self.status_label.setText("Stopped")
    
    def closeEvent(self, event):
        """Ensure proper cleanup when closing"""
        self.stop_processing()
        if self.model_generator and self.model_generator.isRunning():
            self.model_generator.quit()
            self.model_generator.wait()
        event.accept()


def main():
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    
    window = DepthEstimationInterface()
    window.show()
    
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()