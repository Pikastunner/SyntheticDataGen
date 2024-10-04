import sys
import numpy as np
import cv2
from PyQt5.QtWidgets import (QApplication, QMainWindow, QPushButton, QVBoxLayout, QWidget, 
                             QFileDialog, QTableWidget, QHeaderView, QLabel, QSlider, QHBoxLayout, QSpinBox, QGroupBox)
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt

from rembg import remove
from PIL import Image


class ImageProcessingApp(QMainWindow):
    def __init__(self):
        super().__init__()

        # Set up the main window
        self.setWindowTitle("Image Processing App")
        self.setGeometry(100, 100, 1000, 600)

        # Initialize the green threshold values first
        self.lower_green = np.array([35, 20, 30])  # default lower threshold
        self.upper_green = np.array([86, 255, 255])  # default upper threshold
        self.close_kernel_size = 20  # default kernel size for closing


        # Create central widget and layout
        self.central_widget = QWidget(self)
        self.setCentralWidget(self.central_widget)
        self.layout = QVBoxLayout(self.central_widget)

        # Create buttons for loading images
        self.load_rgb_button = QPushButton("Load RGB Images", self)
        self.load_rgb_button.clicked.connect(self.load_rgb_images)
        self.layout.addWidget(self.load_rgb_button)

        self.load_depth_button = QPushButton("Load Depth Images", self)
        self.load_depth_button.clicked.connect(self.load_depth_images)
        self.layout.addWidget(self.load_depth_button)

        # Create a table to display the images
        self.table_widget = QTableWidget(self)
        self.table_widget.setColumnCount(3)  # RGB, Depth, Extracted Object
        self.table_widget.setHorizontalHeaderLabels(["RGB Image", "Depth Image", "Extracted Object"])
        self.layout.addWidget(self.table_widget)

        # Enable scrolling for the table
        self.table_widget.setVerticalScrollMode(QTableWidget.ScrollPerPixel)
        self.table_widget.horizontalHeader().setStretchLastSection(True)

        # Enable horizontal and vertical stretching for better layout management
        self.table_widget.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table_widget.verticalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)

        # Variables to hold images and parameters
        self.rgb_images = []
        self.depth_images = []


    def update_mask_params(self):
        # Update the table with new mask and extraction
        if self.rgb_images:
            self.update_table()

    def load_rgb_images(self):
        # Open file dialog to load multiple RGB images
        filenames, _ = QFileDialog.getOpenFileNames(self, "Open RGB Images", "", "Image Files (*.png *.jpg *.bmp)")
        if filenames:
            self.rgb_images = [cv2.cvtColor(cv2.imread(filename), cv2.COLOR_BGR2RGB) for filename in filenames]
            self.update_table()

    def load_depth_images(self):
        # Open file dialog to load multiple Depth images
        filenames, _ = QFileDialog.getOpenFileNames(self, "Open Depth Images", "", "Image Files (*.png *.jpg *.bmp)")
        if filenames:
            self.depth_images = [cv2.normalize(cv2.imread(filename, cv2.IMREAD_UNCHANGED), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8) for filename in filenames]
            self.update_table()

    def update_table(self):
        # Clear the table and set rows
        self.table_widget.setRowCount(min(len(self.rgb_images), len(self.depth_images)))

        for i in range(min(len(self.rgb_images), len(self.depth_images))):
            # Create mask and extract object with current parameters
            mask = self.create_mask_with_rembg(self.rgb_images[i])
            object_extracted = self.apply_mask(self.rgb_images[i], mask)

            # Display RGB image
            self.display_image_in_table(self.rgb_images[i], i, 0)

            # Display Depth image
            self.display_image_in_table(self.depth_images[i], i, 1, cmap='gray')

            # Display Extracted Object
            self.display_image_in_table(object_extracted, i, 2)

    def display_image_in_table(self, image, row, column, cmap=None):
        # Convert image for QPixmap
        if cmap == 'gray':
            qimage = QImage(image, image.shape[1], image.shape[0], image.strides[0], QImage.Format_Grayscale8)
        else:
            qimage = QImage(image.data, image.shape[1], image.shape[0], QImage.Format_RGB888)

        label_size = 400  # Increased from 200 for better visibility
        pixmap = QPixmap.fromImage(qimage).scaled(label_size, label_size, Qt.KeepAspectRatio)

        # Create a label to display the pixmap
        label = QLabel(self)
        label.setPixmap(pixmap)
        label.setAlignment(Qt.AlignCenter)

        # Set the row height to fit the scaled image
        self.table_widget.setRowHeight(row, pixmap.height())

        # Add the label to the table
        self.table_widget.setCellWidget(row, column, label)

    # Function to create a mask using rembg
    def create_mask_with_rembg(self, rgb_image):
        # Convert the image to a PIL image format for rembg processing
        pil_image = Image.fromarray(rgb_image)
        
        # Use rembg to remove the background
        result_image = remove(pil_image)
        
        # Convert the result back to an OpenCV format (numpy array)
        result_np = np.array(result_image)
        
        # Extract the alpha channel (background removed areas will be transparent)
        mask = result_np[:, :, 3]  # Alpha channel is the fourth channel
        
        return mask

    def apply_mask(self, rgb_image, mask):
        mask_3channel = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
        object_extracted = cv2.bitwise_and(rgb_image, mask_3channel)
        return object_extracted

    def reduce_noise(self, image, kernel_size=(5, 5)):
        kernel = np.ones(kernel_size, np.uint8)
        cleaned_image = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
        kernel_close = np.ones((self.close_kernel_size, self.close_kernel_size), np.uint8)
        cleaned_image = cv2.morphologyEx(cleaned_image, cv2.MORPH_CLOSE, kernel_close)
        return cleaned_image

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ImageProcessingApp()
    window.show()
    sys.exit(app.exec_())
