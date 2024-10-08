import os
import sys
import cv2
import numpy as np
from PyQt5.QtWidgets import (QApplication, QWidget, QLineEdit, QPushButton, QLabel, QVBoxLayout, QHBoxLayout, 
                             QFileDialog, QMessageBox, QListWidget, QMainWindow, QStackedWidget, QSizePolicy)
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import Qt, QFile, QTextStream
import pyrealsense2 as rs
import cv2.aruco as aruco
from rembg import remove
from PIL import Image
import re

from camera import preview_image, capture_and_save_single_frame

from Window_CapturedPhotoReview import CapturedPhotoReviewScreen


# PreviewScreen class for the camera feed preview
class PreviewScreen(QWidget):
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

        self.timer = self.startTimer(100)
        self.current_frame = None  # Store the current frame for saving it later

        self.saved_rgb_image_filenames = []
        self.saved_depth_image_filenames = []

    def blah(self, event):
        color_image, _ = preview_image()
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
            print("photo taken")  # debug message
            self.saved_rgb_image_filenames.append(photo['rgb_image'])
            self.saved_depth_image_filenames.append(photo['depth_image'])

    def keyPressEvent(self, event):
        """Capture the spacebar press to take a photo."""
        if event.key() == Qt.Key_Space:
            self.take_photo()
    
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
        else:
            print("Already on the last page")


# Function to check if camera is connected
def is_camera_connected():
    return True
    # try:
    #     # Create a context object to manage devices
    #     context = rs.context()

    #     # Get a list of connected devices
    #     devices = context.query_devices()

    #     # Check if any devices are connected
    #     if len(devices) > 0:
    #         print(f"Connected devices: {len(devices)}")
    #         return True
    #     else:
    #         print("No RealSense devices connected.")
    #         return False
    # except Exception as e:
    #     print(f"Error while checking devices: {str(e)}")
    #     return False

# Function to load the QSS file
def load_stylesheet(filename):
    file = QFile(filename)
    file.open(QFile.ReadOnly | QFile.Text)
    stream = QTextStream(file)
    return stream.readAll()

# WelcomeScreen class for the initial welcome screen
class WelcomeScreen(QWidget):
    def __init__(self, parent):
        super().__init__()
        self.parent = parent

        # Create the green half
        green_half = QWidget()
        green_half.setObjectName("GreenHalf")

        # Create the content area
        content_area = QWidget()
        layout = QVBoxLayout()

        # Create the text area in the content area
        top_half = QWidget()
        top_layout = QVBoxLayout()
        bottom_half = QWidget()

        label1 = QLabel("Welcome to SyntheticDataGen")
        label1.setObjectName("Label1")

        label2 = QLabel("Before clicking next, plug your camera in.")
        label2.setObjectName("Label2")
        
        label3 = QLabel("Click Next to continue.")
        label3.setObjectName("Label3")

        top_layout.addWidget(label1)
        top_layout.addWidget(label2)
        top_layout.addWidget(label3)
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
        bottom_widget.setObjectName("BottomWidget")
        bottom_layout = QHBoxLayout()
        bottom_widget.setLayout(bottom_layout)
        
        # Create next button with specified size and alignment
        self.next_button = QPushButton("Next")
        self.next_button.setStyleSheet("background-color: #ededed")
        self.next_button.setFixedSize(100, 30)
        self.next_button.setObjectName("NextButton")
        self.next_button.clicked.connect(self.check_camera)
        bottom_layout.addWidget(self.next_button, 0, Qt.AlignRight | Qt.AlignBottom)

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
        else:
            print("Already on the last page")

# Preprocessing Page
class PreprocessingScreen(QWidget):     
    def __init__(self, parent):
        super().__init__()
        self.parent = parent
                
        # Set up the initial container
        title_layout = QVBoxLayout()
        title_area = QWidget()
        title_area.setObjectName("TitleArea")
        # title_area.setStyleSheet("background-color: #d9d9d9;")

        # Working within the initial container
        title_text_layout = QVBoxLayout()
        label = QLabel("Preprocessing")
        label.setObjectName("PreprocessingLabel")
        # label.setStyleSheet("font-size: 18px; margin: 15px;")
        title_text_layout.addWidget(label)
        title_area.setLayout(title_text_layout)

        # Update changes to initial container
        title_layout.addWidget(title_area)

        # Set up the initial container
        preprocessing_results_layout = QVBoxLayout()
        preprocessing_results_area = QWidget()
        preprocessing_results_area.setObjectName("Preprocessing_results_area")
        # preprocessing_results_area.setStyleSheet("background-color: #d9d9d9;")
        

        # Working within the initial container
        preprocessing_area_layout = QHBoxLayout()
        preprocessing_area_layout.setContentsMargins(0, 0, 0, 0)
        preprocessing_area_layout.setSpacing(0)

        background_section = QWidget()
        background_section.setStyleSheet("margin: 15px")
        background_layout = QVBoxLayout()

        background_title = QLabel("Review removed background")
        background_title.setObjectName("")
        background_title.setStyleSheet("font-size: 12px;")

        self.background_image = QLabel()

        self.processed_images = self.convert_images()
        self.image_index = 0
        qimage = self.numpy_to_qimage(self.processed_images[self.image_index])
        self.background_image.setPixmap(QPixmap.fromImage(qimage))
        self.background_image.setScaledContents(True)  # Allow the pixmap to scale with the label
        self.background_image.setGeometry(self.rect())  # Make QLabel cover the whole widget
        self.background_image.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)  # Set size policy


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
        
        graphical_interface_image = QLabel()
        graphical_interface_image.setStyleSheet("background-color: black")
        pixmap = QPixmap("input_images/rgb_image_1.png")
        graphical_interface_image.setPixmap(pixmap)
        graphical_interface_image.setScaledContents(True)  # Allow the pixmap to scale with the label
        graphical_interface_image.setGeometry(self.rect())  # Make QLabel cover the whole widget
        graphical_interface_image.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)  # Set size policy


        graphical_interface_fs = QPushButton("View fullscreen")
        graphical_interface_fs.setStyleSheet("background-color: #ededed")

        graphical_interface_fs.setFixedSize(150, 55)
        graphical_interface_layout.addWidget(graphical_interface_title, 10)
        graphical_interface_layout.addWidget(graphical_interface_image, 86)
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

    def convert_images(self):
        processed_images = []
        rgb_images = self.load_rgb_images()
        depth_images = self.load_depth_images()
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
    
    def load_rgb_images(self):
        folder_path = '../test_images'
        rgb_image_files = self.get_files_starting_with(folder_path, 'rgb_image')
        if rgb_image_files:
            rgb_images = [cv2.cvtColor(cv2.imread(filename), cv2.COLOR_BGR2RGB) for filename in rgb_image_files]
            return rgb_images

    def load_depth_images(self):
        folder_path = '../test_images'
        depth_image_files = self.get_files_starting_with(folder_path, 'depth_image')
        if depth_image_files:
            depth_images = [cv2.normalize(cv2.imread(filename, cv2.IMREAD_UNCHANGED), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8) for filename in depth_image_files]
            return depth_images
        
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
        # self.preview_screen = PreviewScreen(self.central_widget)
        self.captured_photoReview_screen = CapturedPhotoReviewScreen(self.central_widget)
        self.preprocessingScreen = PreprocessingScreen(self.central_widget)

        # Add screens to the stacked widget
        self.central_widget.addWidget(self.welcome_screen)
        # self.central_widget.addWidget(self.preview_screen)
        self.central_widget.addWidget(self.captured_photoReview_screen)
        self.central_widget.addWidget(self.preprocessingScreen)
        
        self.central_widget.setCurrentIndex(0)  # Start with Welcome Screen


if __name__ == '__main__':
    app = QApplication(sys.argv)

    # Load the stylesheet from the QSS file
    stylesheet = load_stylesheet('style.qss')
    app.setStyleSheet(stylesheet)

    main_app = MainApp()
    main_app.show()
    sys.exit(app.exec_())
