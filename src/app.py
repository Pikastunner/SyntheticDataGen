import sys
import cv2
import numpy as np
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QVBoxLayout, QWidget, QPushButton
from PyQt5.QtGui import QImage, QPixmap
import pyrealsense2 as rs

from camera import initialize_camera

class PreviewScreen(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Preview Camera Feed")

        self.image_label = QLabel(self)
        self.image_label.setFixedSize(640, 480)

        layout = QVBoxLayout()
        layout.addWidget(self.image_label)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

        self.pipeline = initialize_camera()
        self.timer = self.startTimer(30)

    
    def timerEvent(self, event):
        color_image = self.capture_frames()
        if color_image is not None:
            # Convert the image to QImage format
            height, width, channel = color_image.shape
            bytes_per_line = 3 * width
            q_image = QImage(color_image.data, width, height, bytes_per_line, QImage.Format_RGB888).rgbSwapped()
            self.image_label.setPixmap(QPixmap.fromImage(q_image))


def main():
    app = QApplication(sys.argv)
    camera_app = PreviewScreen()
    camera_app.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
