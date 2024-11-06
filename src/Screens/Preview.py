
# PreviewScreen class for the camera feed preview
from PyQt5.QtWidgets import (
    QWidget, QPushButton, QLabel, QVBoxLayout,
    QHBoxLayout, QMainWindow
)
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import Qt

from camera import CameraWorker
from camera import is_camera_connected

import cv2
from cv2 import aruco

class PreviewScreen(QMainWindow):
    def __init__(self, parent):
        super().__init__()
        self.parent = parent

        self.setWindowTitle("Preview Camera Feed")

        self.image_label = QLabel(self)
        self.image_label.setFixedSize(640, 480)

        self.take_photo_button = QPushButton("Take Photo", self)
        self.take_photo_button.clicked.connect(self.take_photo)

        bottom_layout = QHBoxLayout()
        self.next_button = QPushButton("Next", self)
        self.next_button.clicked.connect(self.go_to_next_page)
        bottom_layout.addWidget(self.next_button)

        layout = QVBoxLayout()
        layout.addWidget(self.image_label)
        layout.addWidget(self.take_photo_button)

        layout.addLayout(bottom_layout)
        # layout.addWidget(self.next_button)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

        # Create CameraWorker thread
        self.camera_worker = None

        # Initialize the current frame variable
        self.current_frame = (None, None)

        self.saved_rgb_image_filenames = []
        self.saved_depth_image_filenames = []

        # Initialize Aruco parameters
        self.dictionary = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
        self.parameters = aruco.DetectorParameters()
        self.parameters.adaptiveThreshWinSizeMax = 80  # Increase max value for better handling of varying light
        self.parameters.errorCorrectionRate = 1
        # self.parameters.useAruco3Detection = True
        self.parameters.cornerRefinementMethod = aruco.CORNER_REFINE_SUBPIX
        self.parameters.cornerRefinementMaxIterations = 40

        # Detect ArUco markers before background removal
        self.detector = aruco.ArucoDetector(self.dictionary, self.parameters)
        

    def start_camera_worker(self):
        if is_camera_connected():
            # Create CameraWorker thread
            self.camera_worker = CameraWorker()

            # Connect the frameCaptured signal to the update_image method
            self.camera_worker.frameCaptured.connect(self.update_image)

            # Start the camera worker thread
            self.camera_worker.start()

    def update_image(self, rgb_frame, depth_frame):
        """Update the QLabel with the new frame and perform ArUco detection."""
         # Convert to BGR as needed for OpenCV functions
        bgr_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)
        
        # Convert the image to grayscale for marker detection
        gray_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2GRAY)
        
        # Detect ArUco markers
        corners, ids, rejectedImgPoints = self.detector.detectMarkers(gray_frame)
        
        # If markers are detected, annotate the image
        if ids is not None:
            bgr_frame = aruco.drawDetectedMarkers(bgr_frame, corners, ids)  # Draw detected markers on the frame
        
        # Convert BGR back to RGB for displaying
        rgb_image = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)

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
