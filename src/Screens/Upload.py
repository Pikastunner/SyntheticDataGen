from PyQt5.QtWidgets import (QApplication, QWidget, QLineEdit, QPushButton, QLabel, QVBoxLayout, QHBoxLayout, QRadioButton,  QSpacerItem, QSizePolicy,
                             QFileDialog, QMessageBox, QListWidget, QMainWindow, QStackedWidget, QSizePolicy)
from PyQt5.QtCore import Qt, QFile, QTextStream, QThread, pyqtSignal


# UploadScreen class for user to upload image data
class UploadScreen(QWidget):
    def __init__(self, parent):
        super().__init__()
        self.parent = parent

        # Text Section
        text_section = QWidget()
        text_section.setObjectName("UploadTextSection")

        text_section_layout = QVBoxLayout()

        text_area = QWidget()
        
        title = QLabel("Upload RGB and Depth Images")
        title.setObjectName("UploadTitle")

        upload_instruction_title = QLabel("Instructions:")
        upload_instruction_title.setObjectName("UploadNormalText")
        upload_instructions = QLabel("To enhance 3D model generation, please upload RGB (color) images alongside their corresponding depth images. Each RGB image should have an associated depth image file that represents the same scene.")
        upload_instructions.setObjectName("UploadNormalText")


        upload_guideline_title = QLabel("Guidelines:")
        upload_guideline_title.setObjectName("UploadNormalText")
        upload_file_format_info = QLabel("File Format: Supported formats for RGB and Depth images are .jpg, .png, and .bmp.")
        upload_file_format_info.setObjectName("UploadNormalText")

        upload_image_pairing_info = QLabel("Image Pairing: Ensure each RGB image has a matching depth image file uploaded in the same order.")
        upload_image_pairing_info.setObjectName("UploadNormalText")

        text_area_layout = QVBoxLayout()
        text_area_layout.addWidget(title)
        text_area_layout.addWidget(upload_instruction_title)
        text_area_layout.addWidget(upload_instructions)
        text_area_layout.addWidget(upload_guideline_title)
        text_area_layout.addWidget(upload_file_format_info)
        text_area_layout.addWidget(upload_image_pairing_info)
        
        text_area.setLayout(text_area_layout)

        empty_area = QWidget()
        empty_area.setObjectName("UploadTextSection")

        empty_area_layout = QHBoxLayout()

        # RGB Images Upload Area
        upload_rgb_images_area = QWidget()
        upload_rgb_images_area_layout = QVBoxLayout()

        # Spacer above the button
        upload_rgb_images_area_layout.addSpacerItem(QSpacerItem(20, 20, QSizePolicy.Minimum, QSizePolicy.Expanding))

        upload_rgb_images_button = QPushButton("Load RGB Images")
        upload_rgb_images_button.clicked.connect(self.upload_rgb_images)
        upload_rgb_images_area_layout.addWidget(upload_rgb_images_button)

        # Spacer between the button and the label
        upload_rgb_images_area_layout.addSpacerItem(QSpacerItem(20, 20, QSizePolicy.Minimum, QSizePolicy.Expanding))

        self.upload_rgb_images_summary = QLabel("Uploaded 0 images")
        self.upload_rgb_images_summary.setAlignment(Qt.AlignCenter)  # Center the summary text
        upload_rgb_images_area_layout.addWidget(self.upload_rgb_images_summary)

        # Spacer below the label
        upload_rgb_images_area_layout.addSpacerItem(QSpacerItem(20, 20, QSizePolicy.Minimum, QSizePolicy.Expanding))

        upload_rgb_images_area.setLayout(upload_rgb_images_area_layout)

        # Depth Images Upload Area
        upload_depth_images_area = QWidget()
        upload_depth_images_area_layout = QVBoxLayout()

        # Spacer above the button
        upload_depth_images_area_layout.addSpacerItem(QSpacerItem(20, 20, QSizePolicy.Minimum, QSizePolicy.Expanding))

        upload_depth_images_button = QPushButton("Load Depth Images")
        upload_depth_images_button.clicked.connect(self.upload_depth_images)
        upload_depth_images_area_layout.addWidget(upload_depth_images_button)

        # Spacer between the button and the label
        upload_depth_images_area_layout.addSpacerItem(QSpacerItem(20, 20, QSizePolicy.Minimum, QSizePolicy.Expanding))

        self.upload_depth_images_summary = QLabel("Uploaded 0 images")
        self.upload_depth_images_summary.setAlignment(Qt.AlignCenter)  # Center the summary text
        upload_depth_images_area_layout.addWidget(self.upload_depth_images_summary)

        # Spacer below the label
        upload_depth_images_area_layout.addSpacerItem(QSpacerItem(20, 20, QSizePolicy.Minimum, QSizePolicy.Expanding))

        upload_depth_images_area.setLayout(upload_depth_images_area_layout)

        
        empty_area_layout.addWidget(upload_rgb_images_area)
        empty_area_layout.addWidget(upload_depth_images_area)

        empty_area.setLayout(empty_area_layout)

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
        next_button.clicked.connect(self.go_to_next_page)
        
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

        self.saved_depth_image_filenames = []
        self.saved_rgb_image_filenames = []
        
    def go_to_back_page(self):
        current_index = self.parent.currentIndex()
        if current_index > 0:
            self.parent.setCurrentIndex(current_index - 2) 
        else:
            print("Already on the first page")

    def mismatch_images_error(self):
        # Create an error message box
        error_msg = QMessageBox()
        
        # Set critical icon for error message
        error_msg.setIcon(QMessageBox.Critical)
        
        # Set the title of the error message box
        error_msg.setWindowTitle("Image Upload Error")
        
        # Set the detailed text to help the user troubleshoot
        error_msg.setText('<span style="color:#005ace;font-size: 15px;">Mismatched Number of Images</span>')
        error_msg.setInformativeText("Please ensure that each RGB image has a corresponding depth image.")
        
        # Set the standard button to close the message box
        error_msg.setStandardButtons(QMessageBox.Ok)

        # Execute and show the message box
        error_msg.exec_()
        return

    def no_image_uploaded_error(self):
        # Create an error message box
        error_msg = QMessageBox()
        
        # Set critical icon for error message
        error_msg.setIcon(QMessageBox.Critical)
        
        # Set the title of the error message box
        error_msg.setWindowTitle("Image Upload Error")
        
        # Set the detailed text to help the user troubleshoot
        error_msg.setText('<span style="color:#005ace;font-size: 15px;">No Images Uploaded</span>')
        error_msg.setInformativeText("Please upload at least one RGB image and one depth image to proceed.")
        
        # Set the standard button to close the message box
        error_msg.setStandardButtons(QMessageBox.Ok)

        # Execute and show the message box
        error_msg.exec_()
        return

    def go_to_next_page(self):
        current_index = self.parent.currentIndex()
        if (len(self.saved_depth_image_filenames) != len(self.saved_rgb_image_filenames)):
            self.mismatch_images_error()
            return
        if (len(self.saved_depth_image_filenames) == 0 or len(self.saved_rgb_image_filenames) == 0):
            self.no_image_uploaded_error()
            return
        if current_index < self.parent.count() - 1:
            self.parent.setCurrentIndex(current_index + 1)
            next_screen = self.parent.widget(self.parent.currentIndex())
            next_screen.update_variables(self.saved_rgb_image_filenames, self.saved_depth_image_filenames)
        else:
            print("Already on the last page")

    def upload_rgb_images(self):
      self.saved_rgb_image_filenames, _ = QFileDialog.getOpenFileNames(self, "Open RGB Images", "", "Image Files (*.png *.jpg *.bmp)")
      self.upload_rgb_images_summary.setText(f"Uploaded {len(self.saved_rgb_image_filenames)} images")

    def upload_depth_images(self):
      self.saved_depth_image_filenames, _ = QFileDialog.getOpenFileNames(self, "Open Depth Images", "", "Image Files (*.png *.jpg *.bmp)")
      self.upload_depth_images_summary.setText(f"Uploaded {len(self.saved_depth_image_filenames)} images")

