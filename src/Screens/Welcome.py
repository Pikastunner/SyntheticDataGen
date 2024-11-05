from PyQt5.QtWidgets import (QApplication, QWidget, QLineEdit, QPushButton, QLabel, QVBoxLayout, QHBoxLayout, 
                             QFileDialog, QMessageBox, QListWidget, QMainWindow, QStackedWidget, QSizePolicy)
from PyQt5.QtCore import Qt, QFile, QTextStream, QThread, pyqtSignal

from camera import is_camera_connected

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


    def on_load_button_pressed(self):
        self.saved_rgb_image_filenames, _ = QFileDialog.getOpenFileNames(self, "Open RGB Images", "", "Image Files (*.png *.jpg *.bmp)")
        self.saved_depth_image_filenames, _ = QFileDialog.getOpenFileNames(self, "Open Depth Images", "", "Image Files (*.png *.jpg *.bmp)")

        if len(self.saved_rgb_image_filenames) and len(self.saved_depth_image_filenames):
            current_index = self.parent.currentIndex()
            self.parent.setCurrentIndex(current_index + 3)
            next_screen = self.parent.widget(self.parent.currentIndex())
            next_screen.update_variables(self.saved_rgb_image_filenames, self.saved_depth_image_filenames)
