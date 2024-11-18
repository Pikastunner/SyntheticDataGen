from PyQt5.QtWidgets import (QApplication, QWidget, QLineEdit, QPushButton, QLabel, QVBoxLayout, QHBoxLayout, QRadioButton, 
                             QFileDialog, QMessageBox, QListWidget, QMainWindow, QStackedWidget, QSizePolicy)
from PyQt5.QtCore import Qt, QFile, QTextStream, QThread, pyqtSignal

from camera import is_camera_connected

# OptionsScreen class for selecting user's preferences screen
class OptionsScreen(QWidget):
    def __init__(self, parent):
        super().__init__()
        self.parent = parent

        # Text Section
        text_section = QWidget()
        text_section.setObjectName("OptionsTextSection")

        text_section_layout = QVBoxLayout()

        text_area = QWidget()
        
        title = QLabel("Upload or Capture Images")
        title.setObjectName("OptionsTitle")
        options_instructions = QLabel("Choose how you'd like to add images to the program:")
        options_instructions.setObjectName("OptionsNormalText")
        self.load_radio = QRadioButton("Load from Device: Browse and select images and their depth image saved on your device.")
        self.load_radio.setObjectName("OptionsNormalText")
        self.capture_radio = QRadioButton("Capture New Image: Use a plugged in Realsense camera to record image.")
        self.capture_radio.setObjectName("OptionsNormalText")
        self.load_radio.setChecked(True)
        
        text_area_layout = QVBoxLayout()
        text_area_layout.addWidget(title)
        text_area_layout.addWidget(options_instructions)
        text_area_layout.addWidget(self.load_radio)
        text_area_layout.addWidget(self.capture_radio)
        
        text_area.setLayout(text_area_layout)

        empty_area = QWidget()
        empty_area.setObjectName("OptionsTextSection")

        text_section_layout.addWidget(text_area, 40)
        text_section_layout.addWidget(empty_area, 60)

        text_section.setLayout(text_section_layout)

        # Navigation Section
        navigation_area = QWidget(objectName="PreprocessingNavigationArea")
        navigation_buttons_layout = QHBoxLayout(navigation_area)
        spacer = QWidget()
        spacer.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        
        back_button = QPushButton("Back", objectName="BackPageButton")
        back_button.setFixedSize(100, 30)
        back_button.clicked.connect(self.go_to_back_page)

        next_button = QPushButton("Next", objectName="NextPageButton")
        next_button.setFixedSize(100, 30)
        next_button.clicked.connect(self.navigate_to_next_page)
        
        navigation_buttons_layout.addWidget(spacer)
        navigation_buttons_layout.addWidget(back_button)
        navigation_buttons_layout.addSpacing(10)
        navigation_buttons_layout.addWidget(next_button)
        navigation_buttons_layout.setAlignment(Qt.AlignRight | Qt.AlignBottom)

        # Main Layout
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        main_layout.addWidget(text_section, 90)
        main_layout.addWidget(navigation_area, 10)
    
    def navigate_to_next_page(self):
        if self.load_radio.isChecked() == False:   
            self.check_camera()
        else:
            self.go_to_next_page()

    def check_camera(self):
        if is_camera_connected():
            self.go_to_next_page()
            #self.parent.setCurrentIndex(1)  # Go to Camera Preview screen
            preview_screen = self.parent.widget(2)  # Index 1 is the PreviewScreen
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
            if self.load_radio.isChecked() == False:   
              self.parent.setCurrentIndex(current_index + 1)
              print("Capture")
            else:
              self.parent.setCurrentIndex(current_index + 2)
              print("Load")
        else:
            print("Already on the last page")
