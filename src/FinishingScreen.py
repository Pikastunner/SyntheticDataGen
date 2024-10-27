
import os
import sys
import cv2
from PyQt5.QtWidgets import (QWidget, QPushButton, QLabel, QVBoxLayout, QHBoxLayout)
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import Qt
import re

# Preprocessing Page
class FinishingScreen(QWidget):
    def __init__(self, parent):
        super().__init__()
        self.parent = parent

        # Set up the initial container
        text_layout = QVBoxLayout()
        text_area = QWidget()
        text_area.setStyleSheet("background-color: #d9d9d9;")

        # Working within the initial container
        text_container_layout = QVBoxLayout()
        title = QLabel("Synthetic Data Generation Complete")
        title.setStyleSheet(""" 
    font-size: 18px;
    margin: 15px; """)

        self.directory_path = '../input_images'
        location_text = QLabel(f"View data in the following <a href='{self.directory_path}'>directory</a>")
        location_text.setStyleSheet("""    margin-left: 15px;
    margin-right: 15px;
    margin-top: 10px;
    margin-bottom: 10px;
    font-size: 12px;""")
        location_text.setOpenExternalLinks(False)        
        location_text.linkActivated.connect(self.open_directory)

        data_caption = QLabel("Preview of generated data")
        data_caption.setStyleSheet("""    margin-left: 15px;
    margin-right: 20px;
    margin-top: 10px;
    font-size: 12px;""")
        
        text_container_layout.addWidget(title)
        text_container_layout.addWidget(location_text)
        text_container_layout.addWidget(data_caption)
        text_area.setLayout(text_container_layout)

        # Update changes to initial container
        text_layout.addWidget(text_area)

        # Set up the initial container
        generated_data_layout = QHBoxLayout()
        generated_data_area = QWidget()
        generated_data_area.setStyleSheet("background-color: #d9d9d9;")

        # Working within the initial container
        total_layout = QHBoxLayout()
        total_layout.setSpacing(10)
        self.large_image_area = QWidget()
        self.large_image_area.setStyleSheet("background-color: #d9d9d9; margin-left: 15px;")

        large_image_layout = QVBoxLayout()
        large_image_layout.setContentsMargins(0, 0, 0, 0)
        large_image_layout.setSpacing(0)

        self.large_image_region = QLabel()
        self.processed_images = self.load_rgb_images()
        self.image_index = 0

        available_width = int(self.large_image_region.width() * 0.80)
        available_height = int(self.large_image_region.height() * 0.85)
        self.large_image_dim = (available_width, available_height)

        self.apply_image(self.large_image_region, 0, self.large_image_dim)

        navigation_image_region = QWidget()
        navigation_image_region.setStyleSheet("background-color: #d9d9d9;")

        navigation_image_region_layout = QVBoxLayout()
        
        self.image_info = QLabel()
        self.update_image_info()
        self.image_info.setStyleSheet("font-size: 12px;")

        next_button = QPushButton("Next")
        next_button.setStyleSheet("background-color: #ededed")
        next_button.setFixedSize(80, 20)
        next_button.clicked.connect(self.next_image)

        navigation_image_region_layout.addWidget(self.image_info, alignment=Qt.AlignHCenter)
        navigation_image_region_layout.addWidget(next_button, alignment=Qt.AlignHCenter)

        large_image_layout.addWidget(self.large_image_region, 85)
        large_image_layout.addWidget(navigation_image_region, 15)

        navigation_image_region.setLayout(navigation_image_region_layout)

        self.large_image_area.setLayout(large_image_layout)

        self.small_image_area = QWidget()
        self.small_image_area.setStyleSheet("background-color: #d9d9d9; margin-right: 15px;")

        small_image_layout = QVBoxLayout()
        small_image_layout.setContentsMargins(0, 0, 0, 0)
        small_image_layout.setSpacing(0)
        
        self.small_image_region = QWidget()
        self.small_image_region.setStyleSheet("background-color: #d9d9d9;")

        small_image_region_layout = QVBoxLayout()
        small_image_region_layout.setContentsMargins(0, 0, 0, 0)
        small_image_region_layout.setSpacing(10)

        small_available_width = int(self.small_image_region.width() / 3 * 0.80)
        small_available_height = int(self.small_image_region.height() / 3 * 0.85)
        self.small_image_dim = (small_available_width, small_available_height)

        
        self.small_image_one = QLabel()
        self.apply_image(self.small_image_one, 1, self.small_image_dim)
        # self.small_image_one.setStyleSheet("background-color: pink")
        self.small_image_two = QLabel()
        self.apply_image(self.small_image_two, 2, self.small_image_dim)
        # self.small_image_two.setStyleSheet("background-color: green")
        self.small_image_three = QLabel()
        self.apply_image(self.small_image_three, 3, self.small_image_dim)
        # self.small_image_three.setStyleSheet("background-color: yellow")
        
        small_image_region_layout.addWidget(self.small_image_one)
        small_image_region_layout.addWidget(self.small_image_two)
        small_image_region_layout.addWidget(self.small_image_three)

        self.small_image_region.setLayout(small_image_region_layout)

        small_image_layout.setAlignment(Qt.AlignTop)
        small_image_layout.addWidget(self.small_image_region)

        self.small_image_area.setLayout(small_image_layout)

        total_layout.addWidget(self.large_image_area, 75)
        total_layout.addWidget(self.small_image_area, 25)

        generated_data_area.setLayout(total_layout)

        generated_data_layout.addWidget(generated_data_area)


        # Set up the initial container
        bottom_area = QHBoxLayout()      
        bottom_widget = QWidget()
        bottom_widget.setObjectName("BottomWidget")
        bottom_layout = QHBoxLayout()
        bottom_widget.setLayout(bottom_layout)
        
        # Create next button with specified size and alignment
        self.finish_button = QPushButton("Finish")
        self.finish_button.setStyleSheet("background-color: #ededed;")
        self.finish_button.setFixedSize(100, 30)
        self.finish_button.clicked.connect(self.exit_app)
        bottom_layout.addWidget(self.finish_button, 0, Qt.AlignRight | Qt.AlignBottom)

        bottom_area.addWidget(bottom_widget)


        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        main_layout.addLayout(text_layout, 25)
        main_layout.addLayout(generated_data_layout, 70)
        main_layout.addLayout(bottom_area, 5)

        self.setLayout(main_layout) 

    def exit_app(self):
        sys.exit()

    def open_directory(self):
        path = self.parent.widget(self.parent.currentIndex() - 1).directory_input.text()
        # Depending on the OS, open the file explorer to the specified directory
        if os.name == 'nt':  # Windows
            os.startfile(path)

    def load_rgb_images(self):
        folder_path = '../input_images'
        rgb_image_files = self.get_files_starting_with(folder_path, 'rgb_image')
        if rgb_image_files:
            rgb_images = [cv2.cvtColor(cv2.imread(filename), cv2.COLOR_BGR2RGB) for filename in rgb_image_files]
            return rgb_images
    
    def get_files_starting_with(self, folder_path, prefix):
        files = []
        for file in os.listdir(folder_path):
            if re.search(rf"{prefix}_[0-9]", file):
            #file.startswith(prefix):
                files.append(os.path.join(folder_path, file))
        return files

    def numpy_to_qimage(self, img):
        height, width, channel = img.shape
        bytes_per_line = channel * width
        qimage = QImage(img.data, width, height, bytes_per_line, QImage.Format_RGB888)
        return qimage

    def apply_image(self, image: QLabel, index: int, dimensions: tuple):
        qimage = self.numpy_to_qimage(self.processed_images[(self.image_index + index) % len(self.processed_images)])
        image.setPixmap(QPixmap.fromImage(qimage))
        # Calculate the maximum size for the image based on the available space
        available_width = dimensions[0]
        available_height = dimensions[1]
        image_width = qimage.width()
        image_height = qimage.height()
        aspect_ratio = image_width / image_height
        if aspect_ratio > available_width / available_height:
            scaled_width = available_width
            scaled_height = int(scaled_width / aspect_ratio)
        else:
            scaled_height = available_height
            scaled_width = int(scaled_height * aspect_ratio)

        # Set the size of the QLabel and the image
        image.setFixedSize(scaled_width, scaled_height)
        image.setPixmap(QPixmap.fromImage(qimage.scaled(scaled_width, scaled_height)))

    def next_image(self):
        self.image_index = (self.image_index + 1) % len(self.processed_images)
        self.apply_image(self.large_image_region, 0, self.large_image_dim)
        self.apply_image(self.small_image_one, 1, self.small_image_dim)
        self.apply_image(self.small_image_two, 2, self.small_image_dim)
        self.apply_image(self.small_image_three, 3, self.small_image_dim)
        self.update_image_info()  # Update the label text
        return
    
    def update_image_info(self):
        self.image_info.setText(f"Image #{self.image_index + 1} of {len(self.processed_images)}")

