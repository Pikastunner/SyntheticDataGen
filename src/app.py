import sys
import cv2
import numpy as np
from PyQt5.QtWidgets import (QApplication, QWidget, QLineEdit, QPushButton, QLabel, QVBoxLayout, QHBoxLayout, 
                             QFileDialog, QMessageBox, QListWidget, QMainWindow, QStackedWidget, QSizePolicy)
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import Qt, QFile, QTextStream
import pyrealsense2 as rs

from camera import preview_image, capture_and_save_single_frame

# PreviewScreen class for the camera feed preview
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

    def go_to_next_page(self):
        """Switch to the next page."""
        print('next page')


# Function to check if camera is connected
def is_camera_connected():
    try:
        # Create a context object to manage devices
        context = rs.context()

        # Get a list of connected devices
        devices = context.query_devices()

        # Check if any devices are connected
        if len(devices) > 0:
            print(f"Connected devices: {len(devices)}")
            return True
        else:
            print("No RealSense devices connected.")
            return False
    except Exception as e:
        print(f"Error while checking devices: {str(e)}")
        return False

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
            self.parent.setCurrentIndex(1)  # Go to Camera Preview screen
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

# Preprocessing Page
class PreprocessingScreen(QWidget):
    def __init__(self, parent):
        super().__init__()
        self.parent = parent
                
        # Set up the initial container
        title_layout = QVBoxLayout()
        title_area = QWidget()
        title_area.setStyleSheet("background-color: #d9d9d9;")

        # Working within the initial container
        title_text_layout = QVBoxLayout()
        label = QLabel("Preprocessing")
        label.setStyleSheet("font-size: 18px; margin: 15px;")
        title_text_layout.addWidget(label)
        title_area.setLayout(title_text_layout)

        # Update changes to initial container
        title_layout.addWidget(title_area)

        # Set up the initial container
        preprocessing_results_layout = QVBoxLayout()
        preprocessing_results_area = QWidget()
        preprocessing_results_area.setStyleSheet("background-color: #d9d9d9;")
        

        # Working within the initial container
        preprocessing_area_layout = QHBoxLayout()
        preprocessing_area_layout.setContentsMargins(0, 0, 0, 0)
        preprocessing_area_layout.setSpacing(0)

        background_section = QWidget()
        background_section.setStyleSheet("margin: 15px")
        background_layout = QVBoxLayout()

        background_title = QLabel("Review removed background")
        background_title.setStyleSheet("font-size: 12px;")

        background_image = QWidget()
        background_image.setStyleSheet("background-color: black")

        # Center and reduce spacing between background_image_info and background_image_next
        background_image_info = QLabel("Image #1 of 4")
        background_image_info.setStyleSheet("font-size: 12px;")
        background_image_info.setAlignment(Qt.AlignCenter)  # Center the text

        background_image_next = QPushButton("Next")
        background_image_next.setStyleSheet("background-color: #ededed")
        background_image_next.setFixedSize(120, 55)

        # Add a layout to group the info and button together
        center_widget = QWidget()
        center_layout = QVBoxLayout()

        center_layout.addWidget(background_image_info, alignment=Qt.AlignHCenter)
        center_layout.addWidget(background_image_next, alignment=Qt.AlignHCenter)
        center_layout.setContentsMargins(0, 0, 0, 0)
        center_layout.setSpacing(0)
        center_widget.setLayout(center_layout)

        # Add the widgets to the main layout
        background_layout.addWidget(background_title, 10)
        background_layout.addWidget(background_image, 86)
        background_layout.addWidget(center_widget)

        background_section.setLayout(background_layout)

        graphical_interface_section = QWidget()
        graphical_interface_section.setStyleSheet("margin: 15px")
        graphical_interface_layout = QVBoxLayout()
        graphical_interface_title = QLabel("Graphical 3D interface of input image")
        graphical_interface_title.setStyleSheet("font-size: 12px;")
        
        graphical_interface_image = QWidget()
        graphical_interface_image.setStyleSheet("background-color: black")
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
        back_button.clicked.connect(self.go_back)

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
    
    def go_back(self):
        # This method will handle the back navigation
        self.parent.setCurrentIndex(self.parent.currentIndex() - 1)


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
        self.preview_screen = PreviewScreen()

        # Add screens to the stacked widget
        self.central_widget.addWidget(self.welcome_screen)
        self.central_widget.addWidget(self.preview_screen)
        
        self.central_widget.setCurrentIndex(0)  # Start with Welcome Screen


if __name__ == '__main__':
    app = QApplication(sys.argv)

    # Load the stylesheet from the QSS file
    stylesheet = load_stylesheet('style.qss')
    app.setStyleSheet(stylesheet)

    main_app = MainApp()
    main_app.show()
    sys.exit(app.exec_())
