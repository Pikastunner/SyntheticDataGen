import sys
from PyQt5.QtWidgets import (QApplication, QWidget, QPushButton, QLabel, QVBoxLayout, QHBoxLayout, 
                             QFileDialog, QMessageBox, QListWidget, QMainWindow, QStackedWidget)
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import Qt, QFile, QTextStream, QThread, pyqtSignal

from camera import CameraWorker
from camera import is_camera_connected


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

    def go_to_next_page(self):
        """Switch to the next page."""
        print('Next page')

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
