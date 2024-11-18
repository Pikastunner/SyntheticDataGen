
# PreviewScreen class for the camera feed preview

from PyQt5.QtWidgets import (
    QWidget, QPushButton, QLabel, QVBoxLayout,
    QHBoxLayout, QMainWindow
)
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import Qt
import cv2

from camera import CameraWorker, is_camera_connected

import cv2
from cv2 import aruco

class PreviewScreen(QMainWindow):
    def __init__(self, parent):
        super().__init__()
        self.parent = parent

        self.setWindowTitle("Preview Camera Feed")

        # Create label for displaying either RGB or depth frame
        self.image_label = QLabel(self)
        self.image_label.setFixedSize(640, 480)

        # Button to toggle between RGB and depth
        self.toggle_button = QPushButton("Toggle Preview", self)
        self.toggle_button.clicked.connect(self.toggle_image)

        self.take_photo_button = QPushButton("Take Photo", self)
        self.take_photo_button.clicked.connect(self.take_photo)

        # Bottom layout for the navigation button
        bottom_layout = QHBoxLayout()

        self.back_button = QPushButton("Back", self)
        self.back_button.clicked.connect(self.go_to_back_page)
        bottom_layout.addWidget(self.back_button)

        self.next_button = QPushButton("Next", self)
        self.next_button.clicked.connect(self.go_to_next_page)
        bottom_layout.addWidget(self.next_button)
    
        # Main layout with a single image preview and control buttons
        layout = QVBoxLayout()
        layout.addWidget(self.image_label)
        layout.addWidget(self.toggle_button)
        layout.addWidget(self.take_photo_button)
        layout.addLayout(bottom_layout)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

        # Initialize camera worker and current frame variables
        self.camera_worker = None
        self.current_frame = (None, None)
        self.showing_rgb = True  # Start with RGB view
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
            self.camera_worker = CameraWorker()
            self.camera_worker.frameCaptured.connect(self.update_image)
            self.camera_worker.start()

    def update_image(self, rgb_frame, depth_frame):

        """Update the QLabel with the new frame and perform ArUco detection."""
         # Convert to BGR as needed for OpenCV functions
        bgr_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)
        
        # Convert the image to grayscale for marker detection
        gray_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2GRAY)
        
        # Detect ArUco markers
        corners, ids, _ = self.detector.detectMarkers(gray_frame)
        
        # If markers are detected, annotate the image
        if ids is not None:
            bgr_frame = aruco.drawDetectedMarkers(bgr_frame, corners, ids)  # Draw detected markers on the frame
        
        # Convert BGR back to RGB for displaying
        rgb_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        

        """Update the QLabel with the new RGB or depth frame based on the toggle state."""
        self.current_frame = (rgb_frame, depth_frame)
        if self.showing_rgb:
            self.display_rgb_image(rgb_frame)
        else:
            self.display_depth_image(depth_frame)


    def display_rgb_image(self, rgb_frame):
        height, width, channel = rgb_frame.shape
        bytes_per_line = 3 * width
        q_rgb_image = QImage(rgb_frame.tobytes(), width, height, bytes_per_line, QImage.Format_RGB888)
        self.image_label.setPixmap(QPixmap.fromImage(q_rgb_image))

    def display_depth_image(self, depth_frame):
        # Apply colormap for better visualization of the depth frame
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_frame, alpha=0.03), cv2.COLORMAP_JET)
        depth_height, depth_width = depth_colormap.shape[:2]
        q_depth_image = QImage(depth_colormap.tobytes(), depth_width, depth_height, depth_colormap.strides[0], QImage.Format_RGB888)
        self.image_label.setPixmap(QPixmap.fromImage(q_depth_image))

    def toggle_image(self):
        # Toggle display between  RGB and depth frames.
        self.showing_rgb = not self.showing_rgb
        if self.showing_rgb:
            self.toggle_button.setText("Show Depth View")
            self.display_rgb_image(self.current_frame[0])
        else:
            self.toggle_button.setText("Show RGB View")
            self.display_depth_image(self.current_frame[1])

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
            self.parent.setCurrentIndex(current_index + 2)
            next_screen = self.parent.widget(self.parent.currentIndex())
            next_screen.update_variables(self.saved_rgb_image_filenames, self.saved_depth_image_filenames)
        else:
            print("Already on the last page")
