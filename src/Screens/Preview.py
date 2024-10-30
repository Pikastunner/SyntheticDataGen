# preview_screen.py
from PyQt5.QtWidgets import (
    QWidget, QPushButton, QLabel, QVBoxLayout,
    QHBoxLayout, QMainWindow
)
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import Qt
import cv2

from camera import CameraWorker, is_camera_connected

class PreviewScreen(QMainWindow):
    def __init__(self, parent):
        super().__init__()
        self.parent = parent

        self.setWindowTitle("Preview Camera Feed")

        # Create labels for displaying RGB and depth frames
        self.image_label = QLabel(self)
        self.image_label.setFixedSize(640, 480)

        self.depth_image_label = QLabel(self)
        self.depth_image_label.setFixedSize(640, 480)

        self.take_photo_button = QPushButton("Take Photo", self)
        self.take_photo_button.clicked.connect(self.take_photo)

        # Bottom layout for the navigation button
        bottom_layout = QHBoxLayout()
        self.next_button = QPushButton("Next", self)
        self.next_button.clicked.connect(self.go_to_next_page)
        bottom_layout.addWidget(self.next_button)

        # Main layout with RGB and depth image previews
        layout = QVBoxLayout()
        image_layout = QHBoxLayout()
        image_layout.addWidget(self.image_label)
        image_layout.addWidget(self.depth_image_label)
        layout.addLayout(image_layout)
        layout.addWidget(self.take_photo_button)
        layout.addLayout(bottom_layout)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

        # Initialize camera worker and current frame variables
        self.camera_worker = None
        self.current_frame = (None, None)
        self.saved_rgb_image_filenames = []
        self.saved_depth_image_filenames = []

    def start_camera_worker(self):
        if is_camera_connected():
            self.camera_worker = CameraWorker()
            self.camera_worker.frameCaptured.connect(self.update_image)
            self.camera_worker.start()

    def update_image(self, rgb_frame, depth_frame):
        """Update the QLabel with the new RGB and depth frames"""
        # Convert and display the RGB frame
        height, width, channel = rgb_frame.shape
        bytes_per_line = 3 * width
        q_rgb_image = QImage(rgb_frame.tobytes(), width, height, bytes_per_line, QImage.Format_RGB888)
        self.image_label.setPixmap(QPixmap.fromImage(q_rgb_image))

        # Convert and display the depth frame (apply colormap for better visualization)
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_frame, alpha=0.03), cv2.COLORMAP_JET)
        depth_height, depth_width = depth_colormap.shape[:2]
        q_depth_image = QImage(depth_colormap.tobytes(), depth_width, depth_height, depth_colormap.strides[0], QImage.Format_RGB888)
        self.depth_image_label.setPixmap(QPixmap.fromImage(q_depth_image))

        # Store the current frames for saving photos
        self.current_frame = (rgb_frame, depth_frame)

    def take_photo(self):
        photo = self.camera_worker.take_photo()
        if photo:
            print("photo taken")  # debug message
            self.saved_rgb_image_filenames.append(photo['rgb_image'])
            self.saved_depth_image_filenames.append(photo['depth_image'])

    def closeEvent(self, event):
        self.camera_worker.stop()
        event.accept()

    def go_to_back_page(self):
        current_index = self.parent.currentIndex()
        if current_index > 0:
            self.parent.setCurrentIndex(current_index - 1) 
        else:
            print("Already on the first page")

    def go_to_next_page(self):
        current_index = self.parent.currentIndex()
        if current_index < self.parent.count() - 1:
            self.parent.setCurrentIndex(current_index + 1)
            next_screen = self.parent.widget(self.parent.currentIndex())
            next_screen.update_variables()
        else:
            print("Already on the last page")
