import sys
from PyQt5.QtWidgets import (QApplication, QWidget, QPushButton, QLabel, QVBoxLayout, QHBoxLayout, 
                             QFileDialog, QMessageBox, QListWidget, QMainWindow, QStackedWidget)
from PyQt5.QtCore import Qt, QFile, QTextStream
import pyrealsense2 as rs

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

# Welcome Screen
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


# Main Application
class MainApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Synthetic Data Generator")
        self.setFixedSize(700, 650)
        
        self.central_widget = QStackedWidget()
        self.setCentralWidget(self.central_widget)
        
        self.welcome_screen = WelcomeScreen(self.central_widget)
        
        self.central_widget.addWidget(self.welcome_screen)
        
        self.central_widget.setCurrentIndex(0)  # Start with Welcome Screen

if __name__ == '__main__':
    app = QApplication(sys.argv)

    # Load the stylesheet from the QSS file
    stylesheet = load_stylesheet('style.qss')
    app.setStyleSheet(stylesheet)

    main_app = MainApp()
    main_app.show()
    sys.exit(app.exec_())