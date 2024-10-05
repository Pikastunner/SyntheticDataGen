import sys
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QPushButton, QVBoxLayout, QHBoxLayout, QSpacerItem, QSizePolicy, QListWidget, QListWidgetItem
from PyQt5.QtCore import QTimer
from PyQt5.QtGui import QPixmap, QImage

import cv2

from camera import take_photo, initialize_camera, release_camera, preview_image

class CameraPreview(QWidget):
    def __init__(self):
        super().__init__()
        self.photos = []  
        self.cap = None
        self.initUI()

    def initUI(self):

        # Set timer to get live feed of camera
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_camera_feed)
        self.timer.start(16)  

        self.setWindowTitle('Camera preview')
        self.setGeometry(100, 100, 1000, 700)  

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

        self.start_camera()
    
    def start_camera(self):
        self.pipeline = initialize_camera()

    def update_camera_feed(self):
        if self.pipeline is not None:
            color_image, depth_image = preview_image() 
            
            # Convert the color image to QImage
            qt_image = self.convert_to_qimage(color_image)

            # Convert the QImage to a QPixmap and display it
            pixmap = QPixmap.fromImage(qt_image)
            self.camera_view.setPixmap(pixmap.scaled(self.camera_view.size(), Qt.KeepAspectRatio))

    def convert_to_qimage(self, image):
        # Assuming the image is in RGB format
        h, w, ch = image.shape
        bytes_per_line = ch * w
        return QImage(image.data, w, h, bytes_per_line, QImage.Format_RGB888)


    # Populate this function with backend api functionality
    def take_photo(self):
        if self.pipeline is not None:
            color_image, _ = preview_image() 
            
            qt_image = self.convert_to_qimage(color_image)

            pixmap = QPixmap.fromImage(qt_image)
            self.camera_view.setPixmap(pixmap.scaled(self.camera_view.size(), Qt.KeepAspectRatio))

            self.photos.append(pixmap) 
        
    
    def open_photo_preview(self):
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
