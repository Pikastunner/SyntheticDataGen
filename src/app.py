import os
import sys
import cv2
import numpy as np
from PyQt5.QtWidgets import (QApplication, QWidget, QLineEdit, QPushButton, QLabel, QVBoxLayout, QHBoxLayout, 
                             QFileDialog, QMessageBox, QListWidget, QMainWindow, QStackedWidget, QSizePolicy)
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import Qt, QFile, QTextStream, QThread, pyqtSignal
import cv2.aruco as aruco
from rembg import remove
from PIL import Image
import re

import FinishingScreen
from camera import CameraWorker
from camera import is_camera_connected

from Window_CapturedPhotoReview import CapturedPhotoReviewScreen

# PreviewScreen class for the camera feed preview
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
            next_screen.update_variables(self.saved_rgb_image_filenames, self.saved_depth_image_filenames)

        else:
            print("Already on the last page")


# WelcomeScreen class for the initial welcome screen
class WelcomeScreen(QWidget):
    def __init__(self, parent):
        super().__init__()
        self.parent = parent

        # Create the green half
        green_half = QWidget()
        green_half.setStyleSheet("background-color: #84bf3b;")
        # green_half.setObjectName("GreenHalf")

        # Create the content area
        content_area = QWidget()
        layout = QVBoxLayout()

        # Create the text area in the content area
        top_half = QWidget()
        top_layout = QVBoxLayout()
        bottom_half = QWidget()

        label1 = QLabel("Welcome to SyntheticDataGen")
        label1.setStyleSheet("font-weight: bold; font-size: 18px; margin: 15px;")
        # label1.setObjectName("Label1")


        # Create the label with HTML content
        label2 = QLabel('If you wish to capture images, ensure a compatible camera is plugged in.<br><br><br>'
                        'You can read Realsense documentation <a href="https://dev.intelrealsense.com/docs">here</a>.')
        # Enable HTML formatting
        label2.setOpenExternalLinks(True)  # Allow links to open in the default web browser

        label2.setStyleSheet("""margin-left: 15px;
    margin-right: 30px;
    margin-top: 10px;
    margin-bottom: 10px;
    font-size: 12px;""")
        
        
        top_layout.addWidget(label1)
        top_layout.addWidget(label2)
        top_half.setLayout(top_layout)

        layout.addWidget(top_half, 25)
        layout.addWidget(bottom_half, 75)
        content_area.setLayout(layout)

        # Create top area
        top_area = QHBoxLayout()
        top_area.addWidget(green_half, 32)  # Left green area
        top_area.addWidget(content_area, 68)  # Right content area

        # Create bottom content
        bottom_widget = QWidget()
        # bottom_widget.setObjectName("BottomWidget")
        bottom_widget.setStyleSheet("""     background-color: #d9d9d9;
""")
        bottom_layout = QHBoxLayout()
        bottom_widget.setLayout(bottom_layout)

        # Create a horizontal layout for the buttons
        button_layout = QHBoxLayout()

        # Load button
        self.load_button = QPushButton("Load")
        self.load_button.setStyleSheet("background-color: #ededed; margin-right: 5px")
        self.load_button.setToolTip("Start from a pre-exsting set of images.")
        self.load_button.setFixedSize(100, 30)
        self.load_button.setObjectName("LoadButton")
        self.load_button.clicked.connect(self.on_load_button_pressed)
        button_layout.addWidget(self.load_button)

        # Next button
        self.next_button = QPushButton("Capture")
        self.next_button.setStyleSheet("background-color: #ededed;")
        self.next_button.setToolTip("Preview your camera output and capture images.")
        self.next_button.setFixedSize(100, 30)
        self.next_button.setObjectName("NextButton")
        self.next_button.clicked.connect(self.check_camera)
        button_layout.addWidget(self.next_button)

        # Align the button layout to the bottom right
        bottom_layout.addLayout(button_layout)  # Add without alignment
        bottom_layout.setAlignment(button_layout, Qt.AlignRight | Qt.AlignBottom)


        # Create bottom area
        bottom_area = QHBoxLayout()      
        bottom_area.addWidget(bottom_widget)

        # Create the main layout
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        
        main_layout.addLayout(top_area, 90)
        main_layout.addLayout(bottom_area, 10)
        
        self.setLayout(main_layout)
    
    def check_camera(self):
        if is_camera_connected():
            self.go_to_next_page()
            #self.parent.setCurrentIndex(1)  # Go to Camera Preview screen
            preview_screen = self.parent.widget(1)  # Index 1 is the PreviewScreen
            preview_screen.start_camera_worker()
        else:
            # Create an error message box
            error_msg = QMessageBox()
            
            # Set critical icon for error message
            error_msg.setIcon(QMessageBox.Critical)
            
            # Set the title of the error message box
            error_msg.setWindowTitle("Camera Connection Error")
            
            # Set the detailed text to help the user troubleshoot
            error_msg.setText('<span style="color:#005ace;font-size: 15px;">No RealSense camera detected!</span>')
            error_msg.setInformativeText("Please make sure your Intel RealSense camera is plugged into a USB port and try again.")
            
            # Set the standard button to close the message box
            error_msg.setStandardButtons(QMessageBox.Ok)

            # Execute and show the message box
            error_msg.exec_()
        
    def go_to_back_page(self, jump=1):
        current_index = self.parent.currentIndex()
        if current_index > 0:
            self.parent.setCurrentIndex(current_index - jump) 
        else:
            print("Already on the first page")

    def go_to_next_page(self, jump=1):
        current_index = self.parent.currentIndex()
        if current_index < self.parent.count() - 1:
            self.parent.setCurrentIndex(current_index + jump)
        else:
            print("Already on the last page")

    def on_load_button_pressed(self):
        self.saved_rgb_image_filenames, _ = QFileDialog.getOpenFileNames(self, "Open RGB Images", "", "Image Files (*.png *.jpg *.bmp)")
        self.saved_depth_image_filenames, _ = QFileDialog.getOpenFileNames(self, "Open Depth Images", "", "Image Files (*.png *.jpg *.bmp)")

        if len(self.saved_rgb_image_filenames) and len(self.saved_depth_image_filenames):
            current_index = self.parent.currentIndex()
            self.parent.setCurrentIndex(current_index + 3)
            next_screen = self.parent.widget(self.parent.currentIndex())
            next_screen.update_variables(self.saved_rgb_image_filenames, self.saved_depth_image_filenames)


# Preprocessing Page
class PreprocessingScreen(QWidget):  
    def update_variables(self, rgb_filenames, depth_filenames):
        self.processed_images = self.convert_images(rgb_filenames, depth_filenames)
        self.image_index = 0
        qimage = self.numpy_to_qimage(self.processed_images[self.image_index])
        self.background_image.setPixmap(QPixmap.fromImage(qimage))
        self.background_image.setScaledContents(True)  # Allow the pixmap to scale with the label
        self.background_image.setGeometry(self.rect())  # Make QLabel cover the whole widget
        self.background_image.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)  # Set size policy
        self.background_image_info.setText(f"Image #1 of {len(self.processed_images)}")
        
    def __init__(self, parent):
        super().__init__()
        self.parent = parent
                
        # Set up the initial container
        title_layout = QVBoxLayout()
        title_area = QWidget()
        # title_area.setObjectName("TitleArea")
        title_area.setStyleSheet("background-color: #d9d9d9;")

        # Working within the initial container
        title_text_layout = QVBoxLayout()
        label = QLabel("Preprocessing")
        # label.setObjectName("PreprocessingLabel")
        label.setStyleSheet("font-size: 18px; margin: 15px;")
        title_text_layout.addWidget(label)
        title_area.setLayout(title_text_layout)

        # Update changes to initial container
        title_layout.addWidget(title_area)

        # Set up the initial container
        preprocessing_results_layout = QVBoxLayout()
        preprocessing_results_area = QWidget()
        # preprocessing_results_area.setObjectName("Preprocessing_results_area")
        preprocessing_results_area.setStyleSheet("background-color: #d9d9d9;")

        # Working within the initial container
        preprocessing_area_layout = QHBoxLayout()
        preprocessing_area_layout.setContentsMargins(0, 0, 0, 0)
        preprocessing_area_layout.setSpacing(0)

        background_section = QWidget()
        background_section.setStyleSheet("margin: 15px")
        background_layout = QVBoxLayout()

        background_title = QLabel("Review removed background")
        # background_title.setObjectName("")
        background_title.setStyleSheet("font-size: 12px;")

        self.background_image = QLabel()
        self.processed_images = []

        # Center and reduce spacing between background_image_info and background_image_next
        self.background_image_info = QLabel(f"Image #1 of {len(self.processed_images)}")
        self.background_image_info.setStyleSheet("font-size: 12px;")
        self.background_image_info.setAlignment(Qt.AlignCenter)  # Center the text

        background_image_next = QPushButton("Next")
        background_image_next.clicked.connect(self.move_to_next)
        background_image_next.setStyleSheet("background-color: #ededed")
        background_image_next.setFixedSize(120, 55)

        # Add a layout to group the info and button together
        center_widget = QWidget()
        center_layout = QVBoxLayout()

        center_layout.addWidget(self.background_image_info, alignment=Qt.AlignHCenter)
        center_layout.addWidget(background_image_next, alignment=Qt.AlignHCenter)
        center_layout.setContentsMargins(0, 0, 0, 0)
        center_layout.setSpacing(0)
        center_widget.setLayout(center_layout)

        # Add the widgets to the main layout
        background_layout.addWidget(background_title, 10)
        background_layout.addWidget(self.background_image, 86)
        background_layout.addWidget(center_widget)

        background_section.setLayout(background_layout)

        graphical_interface_section = QWidget()
        graphical_interface_section.setStyleSheet("margin: 15px")
        graphical_interface_layout = QVBoxLayout()
        graphical_interface_title = QLabel("Graphical 3D interface of input image")
        graphical_interface_title.setStyleSheet("font-size: 12px;")
        
        graphical_interface_fs = QPushButton("View fullscreen")
        graphical_interface_fs.setStyleSheet("background-color: #ededed")

        graphical_interface_fs.setFixedSize(150, 55)
        graphical_interface_layout.addWidget(graphical_interface_title, 10)
        graphical_interface_layout.addWidget(graphical_interface_fs, 4, alignment=Qt.AlignHCenter)
        
        graphical_interface_section.setLayout(graphical_interface_layout)

        preprocessing_area_layout.addWidget(background_section, 55)
        preprocessing_area_layout.addWidget(graphical_interface_section, 45)

        preprocessing_results_area.setLayout(preprocessing_area_layout)

        # Update changes to initial container
        preprocessing_results_layout.addWidget(preprocessing_results_area)
        
        # Set up the initial container
        directory_saving_layout = QVBoxLayout()
        directory_saving_area = QWidget()
        directory_saving_area.setStyleSheet("background-color: #d9d9d9;")
        directory_saving_area.setContentsMargins(15, 15, 15, 15)

        # Working within the initial container
        directory_text_layout = QVBoxLayout()
        directory_instructions = QLabel("Select a directory to save the synthetic data.")
        directory_instructions.setStyleSheet("font-size: 12px;")

        directory_text_layout.addWidget(directory_instructions)
        directory_saving_area.setLayout(directory_text_layout)

        # Create directory input box and browse button
        self.directory_input = QLineEdit()
        self.directory_input.setFixedHeight(25)  # Set the desired height
        self.directory_input.setStyleSheet("background-color: #ededed; border: none;")

        browse_button = QPushButton("Browse")
        browse_button.setStyleSheet("background-color: #ededed;")
        browse_button.setFixedHeight(25)  # Set the desired height
        browse_button.clicked.connect(self.select_directory)

        # Create layout for input box and button
        directory_input_layout = QHBoxLayout()
        directory_input_layout.addWidget(self.directory_input)
        directory_input_layout.addWidget(browse_button)
        directory_text_layout.addLayout(directory_input_layout)

        # Update changes to initial container
        directory_saving_layout.addWidget(directory_saving_area)

        navigation_layout = QHBoxLayout()
        navigation_area = QWidget()
        navigation_area.setStyleSheet("background-color: #d9d9d9;")

        navigation_buttons_layout = QHBoxLayout()

        # Spacer to shift the buttons to the right
        spacer = QWidget()
        spacer.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        navigation_buttons_layout.addWidget(spacer)

        back_button = QPushButton("Back")
        back_button.setFixedSize(100, 30)
        back_button.setStyleSheet("background-color: #ededed;")
        back_button.clicked.connect(self.go_to_back_page)

        next_button = QPushButton("Next")
        next_button.setFixedSize(100, 30)
        next_button.setStyleSheet("background-color: #ededed;")
        next_button.clicked.connect(self.go_to_next_page)

        # Add the buttons to the layout with a small gap between them
        navigation_buttons_layout.addWidget(back_button)
        navigation_buttons_layout.addSpacing(10)  # Set the gap between the buttons
        navigation_buttons_layout.addWidget(next_button)

        # Align buttons to the right and bottom
        navigation_buttons_layout.setAlignment(Qt.AlignRight | Qt.AlignBottom)

        navigation_area.setLayout(navigation_buttons_layout)

        navigation_layout.addWidget(navigation_area)
        
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        
        main_layout.addLayout(title_layout, 10)
        main_layout.addLayout(preprocessing_results_layout, 63)
        main_layout.addLayout(directory_saving_layout, 17)
        main_layout.addLayout(navigation_layout, 10)
        

        self.setLayout(main_layout)
    
    def view_3d_interface(self):
        # Open fullscreen 3D preview (placeholder)
        QMessageBox.information(self, "3D Interface", "Viewing 3D interface.")
    
    def select_directory(self):
        directory = QFileDialog.getExistingDirectory(self, "Select Directory")
        if directory:
            self.directory_input.setText(directory)
            # QMessageBox.information(self, "Directory Selected", f"Data will be saved to {directory}")
    
    def go_to_complete(self):
        pass
        # generate_3d_mesh()  # Simulate 3D mesh generation
        # self.parent.setCurrentIndex(4)

    def move_to_next(self):
        # Increment the index and wrap around if it exceeds the number of processed images
        self.image_index = (self.image_index + 1) % len(self.processed_images)
        self.display_current_image()  # Call to update the image display
        self.update_image_info()  # Update the label text

    def update_image_info(self):
        # Update the QLabel with the current image index
        self.background_image_info.setText(f"Image #{self.image_index + 1} of {len(self.processed_images)}")
    
    def display_current_image(self):
        if self.processed_images:  # Ensure there are images to display
            current_image = self.processed_images[self.image_index]
            qimage = self.numpy_to_qimage(current_image)
            self.background_image.setPixmap(QPixmap.fromImage(qimage))
            self.background_image.setScaledContents(True)  # Optional: Scale to fit the label
            self.background_image.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)  # Set size policy

    def convert_images(self, rgb_filenames, depth_filenames):
        processed_images = []
        rgb_images = self.load_rgb_images(rgb_filenames)
        depth_images = self.load_depth_images(depth_filenames)
        for i in range(min(len(rgb_images), len(depth_images))):
            # Create mask and extract object with current parameters
            mask = self.create_mask_with_rembg(rgb_images[i])
            object_extracted = self.apply_mask(rgb_images[i], mask)
            processed_images.append(object_extracted)
        return processed_images

    def numpy_to_qimage(self, img):
        height, width, channel = img.shape
        bytes_per_line = channel * width
        qimage = QImage(img.data, width, height, bytes_per_line, QImage.Format_RGB888)
        return qimage

    # Function to create a mask using rembg
    def create_mask_with_rembg(self, rgb_image):
        # Convert the image to a PIL image format for rembg processing
        pil_image = Image.fromarray(rgb_image)
        
        # Use rembg to remove the background
        result_image = remove(pil_image)
        
        # Convert the result back to an OpenCV format (numpy array)
        result_np = np.array(result_image)
        
        # Extract the alpha channel (background removed areas will be transparent)
        mask = result_np[:, :, 3]  # Alpha channel is the fourth channel

        # Convert RGB image to grayscale for ArUco detection
        gray_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)

        # Define the dictionary of ArUco markers
        dictionary = aruco.getPredefinedDictionary(aruco.DICT_4X4_250)
        parameters = aruco.DetectorParameters()
        detector = aruco.ArucoDetector(dictionary, parameters)
        corners, _, _ = detector.detectMarkers(gray_image)

        # Create a mask for detected ArUco markers
        if corners: 
            aruco_mask = np.zeros_like(mask)
            for corner in corners:
                cv2.fillConvexPoly(aruco_mask, corner[0].astype(int), 255)
            # Combine the masks using a logical OR operation
            combined_mask = cv2.bitwise_or(mask, aruco_mask)
            return combined_mask
        else:
            # If no ArUco markers are detected, return the original mask
            return mask

    def get_files_starting_with(self, folder_path, prefix):
        files = []
        for file in os.listdir(folder_path):
            if re.search(rf"{prefix}_[0-9]", file):
            #file.startswith(prefix):
                files.append(os.path.join(folder_path, file))
        return files
    
    def apply_mask(self, rgb_image, mask):
        mask_3channel = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
        object_extracted = cv2.bitwise_and(rgb_image, mask_3channel)
        return object_extracted
    
    def load_rgb_images(self, rgb_filenames=None):
        folder_path = './input_images/' if not rgb_filenames else ''
        rgb_images = [cv2.cvtColor(cv2.imread(f"{folder_path}{filename}"), cv2.COLOR_BGR2RGB) for filename in rgb_filenames]
        return rgb_images

    def load_depth_images(self, depth_filenames=None):
        folder_path = './input_images/' if not depth_filenames else ''
        depth_images = [cv2.normalize(cv2.imread(f"{folder_path}{filename}", cv2.IMREAD_UNCHANGED), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8) for filename in depth_filenames]
        return depth_images
        
    def go_to_back_page(self):
        current_index = self.parent.currentIndex()
        if current_index > 0:
            self.parent.setCurrentIndex(current_index - 1) 
        else:
            print("Already on the first page")

    def go_to_next_page(self):
        if (len(self.directory_input.text()) == 0):
            error_msg = QMessageBox()
            
            # Set critical icon for error message
            error_msg.setIcon(QMessageBox.Critical)
            
            # Set the title of the error message box
            error_msg.setWindowTitle("Empty Path Error")
            
            # Set the detailed text to help the user troubleshoot
            error_msg.setText('<span style="color:#005ace;font-size: 15px;">No Path Input!</span>')
            error_msg.setInformativeText("Please make sure you specify a path")
            
            # Set the standard button to close the message box
            error_msg.setStandardButtons(QMessageBox.Ok)

            # Execute and show the message box
            error_msg.exec_()
            return
        current_index = self.parent.currentIndex()
        if current_index < self.parent.count() - 1:
            self.parent.setCurrentIndex(current_index + 1)
        else:
            print("Already on the last page")
    
# Main Application
class MainApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Synthetic Data Generator")
        self.setFixedSize(700, 650)
        
        self.central_widget = QStackedWidget()
        self.setCentralWidget(self.central_widget)
        
        # Welcome screen and preview screen
        self.welcome_screen = WelcomeScreen(self.central_widget)
        self.preview_screen = PreviewScreen(self.central_widget)
        self.captured_photoReview_screen = CapturedPhotoReviewScreen(self.central_widget)
        self.preprocessingScreen = PreprocessingScreen(self.central_widget)
        # self.finishing_screen = FinishingScreen(self.central_widget)

        # Add screens to the stacked widget
        self.central_widget.addWidget(self.welcome_screen)
        self.central_widget.addWidget(self.preview_screen)
        self.central_widget.addWidget(self.captured_photoReview_screen)
        self.central_widget.addWidget(self.preprocessingScreen)
        # self.central_widget.addWidget(self.finishing_screen)
        
        self.central_widget.setCurrentIndex(0)  # Start with Welcome Screen


import os

if __name__ == '__main__':
    app = QApplication(sys.argv)

    with open("src/style.qss","r") as fh:
        app.setStyleSheet(fh.read())

    main_app = MainApp()
    main_app.show()
    sys.exit(app.exec_())
