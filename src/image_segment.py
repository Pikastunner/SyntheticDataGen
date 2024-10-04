import sys
import numpy as np
import cv2
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QVBoxLayout, QWidget, QFileDialog, QTableWidget, QHeaderView, QLabel
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt

class ImageProcessingApp(QMainWindow):
    def __init__(self):
        super().__init__()

        # Set up the main window
        self.setWindowTitle("Image Processing App")
        self.setGeometry(100, 100, 1000, 600)

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


        # Variables to hold images
        self.rgb_images = []
        self.depth_images = []

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
            # Create mask and extract object
            mask = self.create_mask_from_rgb(self.rgb_images[i])
            mask_cleaned = self.remove_mask_boundary_objects(self.reduce_noise(mask, kernel_size=(9, 9)))
            object_extracted = self.apply_mask(self.rgb_images[i], mask_cleaned)

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
        
        # Dynamically calculate the scaling based on the original image size
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


    def create_mask_from_rgb(self, rgb_image):
        hsv = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2HSV)
        lower_green = np.array([35, 20, 30])
        upper_green = np.array([86, 255, 255])
        mask = cv2.inRange(hsv, lower_green, upper_green)
        mask = cv2.bitwise_not(mask)
        return mask

    def apply_mask(self, rgb_image, mask):
        mask_3channel = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
        object_extracted = cv2.bitwise_and(rgb_image, mask_3channel)
        return object_extracted

    def reduce_noise(self, image, kernel_size=(5, 5)):
        kernel = np.ones(kernel_size, np.uint8)
        cleaned_image = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
        kernel_close = np.ones((20, 20), np.uint8)
        cleaned_image = cv2.morphologyEx(cleaned_image, cv2.MORPH_CLOSE, kernel_close)
        return cleaned_image

    def remove_mask_boundary_objects(self, mask):
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
        output_mask = np.zeros_like(mask)
        for i in range(1, num_labels):
            if stats[i, cv2.CC_STAT_LEFT] == 0 or stats[i, cv2.CC_STAT_TOP] == 0 or \
               stats[i, cv2.CC_STAT_LEFT] + stats[i, cv2.CC_STAT_WIDTH] == mask.shape[1] or \
               stats[i, cv2.CC_STAT_TOP] + stats[i, cv2.CC_STAT_HEIGHT] == mask.shape[0]:
                continue
            output_mask[labels == i] = 255
        return output_mask

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ImageProcessingApp()
    window.show()
    sys.exit(app.exec_())
