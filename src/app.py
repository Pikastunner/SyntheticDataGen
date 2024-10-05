import sys
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QPushButton, QVBoxLayout, QHBoxLayout, QSpacerItem, QSizePolicy, QGridLayout, QListWidget, QListWidgetItem
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap, QImage
import cv2
import numpy as np
import os

from camera import take_photo  

class CameraPreview(QWidget):
    def __init__(self):
        super().__init__()
        self.photos = []  
        self.initUI()

    def initUI(self):

        self.setWindowTitle('Camera preview')
        self.setGeometry(100, 100, 1000, 700)  # Increased window size


        main_layout = QVBoxLayout()

        camera_label = QLabel('Camera preview')

        instruction_label = QLabel('Press SPACE to take a photo\nMake sure to take multiple shots at different angles and ensure that the background is green')

        self.camera_view = QLabel(self)
        self.camera_view.setStyleSheet("background-color: black;")
        self.camera_view.setFixedSize(800, 500)  

        button_layout = QHBoxLayout()

        take_photo_button = QPushButton('Take Photo (SPACE)', self)
        take_photo_button.clicked.connect(self.take_photo)
        button_layout.addWidget(take_photo_button)

        spacer = QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)
        button_layout.addItem(spacer)

        # Back button
        back_button = QPushButton('Back', self)
        button_layout.addWidget(back_button)

        # Next button
        next_button = QPushButton('Next', self)
        next_button.clicked.connect(self.open_photo_preview)
        button_layout.addWidget(next_button)

        main_layout.addWidget(camera_label)
        main_layout.addWidget(instruction_label)
        main_layout.addWidget(self.camera_view)
        main_layout.addLayout(button_layout)

        self.setLayout(main_layout)

    def take_photo(self):
        take_photo_output = take_photo()
        
        

    def open_photo_preview(self):
        # Open the preview window and pass the list of photos
        self.preview_window = PhotoPreview(self.photos)
        self.preview_window.show()

class PhotoPreview(QWidget):
    def __init__(self, photos):
        super().__init__()
        self.photos = photos
        self.initUI()

    def initUI(self):
        # Set window title and larger size
        self.setWindowTitle('Photo Preview')
        self.setGeometry(150, 150, 1000, 700)  

        main_layout = QVBoxLayout()

        photo_list = QListWidget(self)

        for photo in self.photos:
            item = QListWidgetItem()
            item.setSizeHint(photo.size())
            photo_label = QLabel()
            photo_label.setPixmap(photo)
            photo_list.addItem(item)
            photo_list.setItemWidget(item, photo_label)

        main_layout.addWidget(photo_list)

        self.setLayout(main_layout)

def main():
    app = QApplication(sys.argv)
    ex = CameraPreview()
    ex.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
