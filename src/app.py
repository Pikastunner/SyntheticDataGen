import sys
import cv2
import numpy as np
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QVBoxLayout, QWidget, QPushButton
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import Qt
import pyrealsense2 as rs

from camera import preview_image, capture_and_save_single_frame

class PreviewScreen(QMainWindow):
    def __init__(self):
        super().__init__()
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

        self.timer = self.startTimer(100)
        self.current_frame = None  # Store the current frame for saving it later

        self.saved_rgb_image_filenames = []
        self.saved_depth_image_filenames = []

    def timerEvent(self, event):
        color_image = preview_image()
        if color_image is not None:
            self.current_frame = color_image  # Store the current frame
            # Convert the image to QImage format
            height, width, channel = color_image.shape
            bytes_per_line = 3 * width
            q_image = QImage(color_image.data, width, height, bytes_per_line, QImage.Format_RGB888).rgbSwapped()
            self.image_label.setPixmap(QPixmap.fromImage(q_image))

    def take_photo(self):
        """Save the current frame as an image file."""
        photo = capture_and_save_single_frame("input_images")
        if photo is not None:
            self.saved_rgb_image_filenames.append(photo['rgb_image'])
            self.saved_depth_image_filenames.append(photo['depth_image'])

    def keyPressEvent(self, event):
        """Capture the spacebar press to take a photo."""
        if event.key() == Qt.Key_Space:
            self.take_photo()

    def go_to_next_page(self):
        """Switch to the next page."""
        print('next page')


def main():
    app = QApplication(sys.argv)
    camera_app = PreviewScreen()
    camera_app.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
