# PreviewScreen class for the camera feed preview
from PyQt5.QtWidgets import (QWidget, QPushButton, QLabel, QVBoxLayout, QMainWindow)
from PyQt5.QtGui import QImage, QPixmap

from camera import CameraWorker
from camera import is_camera_connected

class PreviewScreen(QMainWindow):
    def __init__(self, parent):
        super().__init__()
        self.parent = parent

        self.setWindowTitle("Preview Camera Feed")

        self.image_label = QLabel(self)
        self.image_label.setFixedSize(640, 480)

        self.take_photo_button = QPushButton("Take Photo", self)
        self.take_photo_button.clicked.connect(self.take_photo)

        self.next_button = QPushButton("Next", self)
        self.next_button.clicked.connect(self.go_to_next_page)

        layout = QVBoxLayout()
        layout.addWidget(self.image_label)
        layout.addWidget(self.take_photo_button)
        layout.addWidget(self.next_button)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

        # Create CameraWorker thread
        self.camera_worker = None

        # Initialize the current frame variable
        self.current_frame = (None, None)

        self.saved_rgb_image_filenames = []
        self.saved_depth_image_filenames = []

    def start_camera_worker(self):
        if is_camera_connected():
            # Create CameraWorker thread
            self.camera_worker = CameraWorker()

            # Connect the frameCaptured signal to the update_image method
            self.camera_worker.frameCaptured.connect(self.update_image)

            # Start the camera worker thread
            self.camera_worker.start()

    def update_image(self, rgb_frame, depth_frame):
        """Update the QLabel with the new frame"""
        # Convert BGR to RGB
        rgb_image = rgb_frame

        # Convert the image to QImage format
        height, width, channel = rgb_image.shape
        bytes_per_line = 3 * width
        
        # Convert memoryview to bytes before passing to QImage
        q_image = QImage(rgb_image.tobytes(), width, height, bytes_per_line, QImage.Format_RGB888)
        self.image_label.setPixmap(QPixmap.fromImage(q_image))

        # Store the current frame for photo capture
        self.current_frame = (rgb_frame, depth_frame)

    def take_photo(self):
        photo = self.camera_worker.take_photo()
        if photo:
            print("photo taken")  # debug message
            self.saved_rgb_image_filenames.append(photo['rgb_image'])
            self.saved_depth_image_filenames.append(photo['depth_image'])

    def closeEvent(self, event):
        """Handle the window close event and stop the camera thread."""
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
