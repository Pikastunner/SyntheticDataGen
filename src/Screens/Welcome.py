from PyQt5.QtWidgets import (QApplication, QWidget, QLineEdit, QPushButton, QLabel, QVBoxLayout, QHBoxLayout, 
                             QFileDialog, QMessageBox, QListWidget, QMainWindow, QStackedWidget, QSizePolicy)
from PyQt5.QtCore import Qt, QFile, QTextStream, QThread, pyqtSignal

# WelcomeScreen class for the initial welcome screen
class WelcomeScreen(QWidget):
    def __init__(self, parent):
        super().__init__()
        self.parent = parent

        # Create the green half
        green_half = QWidget(objectName="GreenHalf")

        # Create the content area
        content_area = QWidget()
        content_area.setObjectName("ContentArea")
        layout = QVBoxLayout()

        # Create the text area in the content area
        top_half = QWidget()
        top_layout = QVBoxLayout()
        bottom_half = QWidget()

        # Add text to the Welcome Screen
        label1 = QLabel("Welcome to SyntheticDataGen", objectName="Label1")
        label2 = QLabel('If you wish to capture images, ensure a compatible camera is plugged in.<br><br><br>'
                        'You can read Realsense documentation <a href="https://dev.intelrealsense.com/docs">here</a>.', 
                        objectName="Label2")
        label2.setOpenExternalLinks(True) # Allow the link to ework
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
        bottom_widget = QWidget(objectName="BottomWidget")
        bottom_layout = QHBoxLayout()
        bottom_widget.setLayout(bottom_layout)
        
        # Create a horizontal layout for the buttons
        button_layout = QHBoxLayout()


        # Next button
        next_button = QPushButton("Next")
        next_button.setFixedSize(100, 30)
        next_button.clicked.connect(self.go_to_next_page)
        button_layout.addWidget(next_button)
        
        button_layout.setSpacing(16)  # Set spacing to 10 pixels

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

if __name__ == "__main__":
    from PyQt5.QtGui import QIcon
    import sys

    class MainApp(QMainWindow):
        def __init__(self):
            super().__init__()
            self.setWindowTitle("Synthetic Data Generator")
            self.setFixedSize(700, 650)
            self.setWindowIcon(QIcon("./src/Icons/app_icon.svg")) 

            
            # Central widget for layout management
            self.central_widget = QWidget()
            self.setCentralWidget(self.central_widget)
            
            # Main layout that will contain the stacked widget
            self.main_layout = QVBoxLayout(self.central_widget)
            self.main_layout.setContentsMargins(0, 0, 0, 0)  # Remove margins around the layout
            self.main_layout.setSpacing(0)  # Remove spacing between elements in the layout
            
            # Main widget for switching between scenes
            self.stacked_widget = QStackedWidget()
            self.stacked_widget.setContentsMargins(0, 0, 0, 0) 
            self.main_layout.addWidget(self.stacked_widget)
            
            # Add screens
            self.stacked_widget.addWidget(WelcomeScreen(self.stacked_widget))
            
            self.stacked_widget.setCurrentIndex(0)

    app = QApplication(sys.argv)

    with open("src/Stylesheets/style_light.qss", "r") as fh:
        app.setStyleSheet(fh.read())

    main_app = MainApp()
    main_app.show()
    sys.exit(app.exec_())