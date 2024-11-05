import sys
import numpy as np
import cv2
import cv2.aruco as aruco
from PyQt5.QtWidgets import (QApplication, QWidget, QPushButton, QVBoxLayout, QTableWidget, 
                             QHeaderView, QLabel, QFileDialog)
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt, pyqtSignal
from rembg import remove
from PIL import Image
import os

from params import *

class BackgroundRemover(QWidget):
    '''
    This PyQt widget takes in several images and returns the images without their backgrounds.
    '''

    update_complete = pyqtSignal(list, list, list)  # Updated to pass RGB and depth images


    def __init__(self):
        super().__init__()

        # Create layout
        self.layout = QVBoxLayout(self)

        # Create buttons for loading images with indicators
        self.load_rgb_button = QPushButton("Load RGB Images ❌", self)
        self.load_rgb_button.clicked.connect(self.load_rgb_images)
        self.layout.addWidget(self.load_rgb_button)

        self.load_depth_button = QPushButton("Load Depth Images ❌", self)
        self.load_depth_button.clicked.connect(self.load_depth_images)
        self.layout.addWidget(self.load_depth_button)

        # Create a table to display the images
        self.table_widget = QTableWidget(self)
        self.table_widget.setColumnCount(4)  # RGB, Depth, Extracted Object
        self.table_widget.setHorizontalHeaderLabels(["RGB Image", "Depth Image", "Preview", "Markers"])
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
            self.load_rgb_button.setText("Load RGB Images ✔️")
            self.update_table()
        else:
            self.load_rgb_button.setText("Load RGB Images ❌")

    def load_depth_images(self):
        # Open file dialog to load multiple Depth images
        filenames, _ = QFileDialog.getOpenFileNames(self, "Open Depth Images", "", "Image Files (*.png *.jpg *.bmp)")
        if filenames:
            self.depth_images = [cv2.normalize(cv2.imread(filename, cv2.IMREAD_UNCHANGED), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8) for filename in filenames]
            self.load_depth_button.setText("Load Depth Images ✔️")
            self.update_table()
        else:
            self.load_depth_button.setText("Load Depth Images ❌")

    def set_depths(self, filenames):
        self.depth_images = [cv2.normalize(cv2.imread(filename, cv2.IMREAD_UNCHANGED), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8) for filename in filenames]
        self.update_table()

    def set_rgbs(self, filenames):
        self.rgb_images = [cv2.cvtColor(cv2.imread(filename), cv2.COLOR_BGR2RGB) for filename in filenames]
        self.update_table()

    def update_table(self):
        # Clear the table and set rows
        self.table_widget.setRowCount(min(len(self.rgb_images), len(self.depth_images)))

        extracted_objects = []  # List to store extracted objects
        aruco_datas = []  # List to store aruco information

        def estimatePoseSingleMarkers(corners, marker_size, mtx, distortion):
            marker_points = np.array([[-marker_size / 2, marker_size / 2, 0],
                                    [marker_size / 2, marker_size / 2, 0],
                                    [marker_size / 2, -marker_size / 2, 0],
                                    [-marker_size / 2, -marker_size / 2, 0]], dtype=np.float32)
            rvecs = []
            tvecs = []
            for c in corners:
                _, R, t = cv2.solvePnP(marker_points, c, mtx, distortion, False, cv2.SOLVEPNP_EPNP)
                rvecs.append(R)
                tvecs.append(t)
            return rvecs, tvecs
        

        emitted_depths = []

        for i in range(min(len(self.rgb_images), len(self.depth_images))):
            # Create masks for the object and ArUco markers
            object_mask, aruco_mask, aruco_data = self.create_mask_with_rembg(self.rgb_images[i])

            corners = aruco_data[0]
            ids = aruco_data[1]

            if ids is None or len(ids) <= 1:
                continue

            emitted_depths.append(self.depth_images[i])
            aruco_datas.append(aruco_data)

            # Extract object without ArUco markers
            object_extracted = self.apply_mask(self.rgb_images[i], object_mask)

            # Extract ArUco markers to see whether they have been detected properly
            aruco_extracted = self.apply_mask(self.rgb_images[i], aruco_mask)

            rvecs, tvecs = estimatePoseSingleMarkers(corners, 0.02, camera_matrix(), dist_coeffs())

            # Draw axes for each detected marker
            for rvec, tvec in zip(rvecs, tvecs):
                # Draw the axes
                aruco_extracted = cv2.drawFrameAxes(aruco_extracted, camera_matrix(), dist_coeffs(), rvec, tvec, 0.03)  # 0.1 is the length of the axes

            aruco_extracted = aruco.drawDetectedMarkers(aruco_extracted, corners, ids)

            # Store the extracted object, ArUco markers, and depth for later use
            extracted_objects.append(object_extracted)
            
            # Display images in row
            self.display_image_in_table(self.rgb_images[i], i, 0)
            self.display_image_in_table(self.depth_images[i], i, 1, cmap='gray')
            self.display_image_in_table(object_extracted, i, 2)
            self.display_image_in_table(aruco_extracted, i, 3)

        # Emit extracted objects and markers
        if len(self.depth_images) and len(self.rgb_images):
            self.update_complete.emit(extracted_objects, emitted_depths, aruco_datas)  # Emit with extracted objects


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
    
    def create_mask_with_rembg(self, rgb_image):
        # Convert RGB image to grayscale for ArUco detection
        gray_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)

        # Define the dictionary of ArUco markers
        dictionary = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)

        # Adjust DetectorParameters for more sensitivity
        parameters = aruco.DetectorParameters()
        parameters.adaptiveThreshWinSizeMax = 75  # Increase max value for better handling of varying light
        parameters.useAruco3Detection = True
        parameters.cornerRefinementMethod = aruco.CORNER_REFINE_SUBPIX
        parameters.cornerRefinementMaxIterations = 40

        # Detect ArUco markers before background removal
        detector = aruco.ArucoDetector(dictionary, parameters)
        corners, ids, rejected = detector.detectMarkers(gray_image)

        if ids is not None:
            corners, ids, _, recovered = detector.refineDetectedMarkers(gray_image, board=aruco_board(), detectedCorners=corners, detectedIds=ids, rejectedCorners=rejected,cameraMatrix=camera_matrix(),distCoeffs=dist_coeffs())
            print(f"{len(ids)} aruco markers found")


        # Create a mask for detected ArUco markers
        aruco_mask = np.zeros(rgb_image.shape[:2], dtype=np.uint8)
        if corners:
            for corner in corners:
                cv2.fillConvexPoly(aruco_mask, corner[0].astype(int), 255)

        # Dilate the ArUco marker mask to grow it slightly
        kernel_size = 5  # Adjust size based on the thickness of outlines
        dilation_kernel = np.ones((kernel_size, kernel_size), np.uint8)
        dilated_aruco_mask = cv2.dilate(aruco_mask, dilation_kernel, iterations=1)

        # Convert the image to a PIL image format for rembg processing
        pil_image = Image.fromarray(rgb_image)

        # Use rembg to remove the background
        result_image = remove(pil_image)

        # Convert the result back to an OpenCV format (numpy array)
        result_np = np.array(result_image)

        # Extract the alpha channel (background removed areas will be transparent)
        object_mask = result_np[:, :, 3]  # Alpha channel is the fourth channel

        # Invert the dilated ArUco mask so that the area around the markers is excluded from the object
        inverted_dilated_aruco_mask = cv2.bitwise_not(dilated_aruco_mask)

        # Apply the inverted dilated mask to the object mask to exclude the markers and their surroundings
        refined_object_mask = cv2.bitwise_and(object_mask, inverted_dilated_aruco_mask)

        # Return the refined object mask, the aruco mask and the aruco information
        return refined_object_mask, aruco_mask, (corners, ids)


    def apply_mask(self, rgb_image, mask):
        mask_3channel = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
        object_extracted = cv2.bitwise_and(rgb_image, mask_3channel)
        return object_extracted


class GenerateWidget(QWidget):
    def __init__(self):
        super().__init__()

        # Create layout
        self.layout = QVBoxLayout(self)

        # Create a button with "Generate" label
        self.generate_button = QPushButton("Generate", self)
        self.generate_button.clicked.connect(self.on_generate)  # Connect button to the slot method

        # Add the button to the layout
        self.layout.addWidget(self.generate_button)

        # Variables to hold images
        self.rgb_images = []
        self.depth_images = []

    def set_images(self, rgb_images, depth_images):
        # Store the images received from BackgroundRemover
        self.rgb_images = rgb_images
        self.depth_images = depth_images
        print(f"Received {len(rgb_images)} RGB images and {len(depth_images)} Depth images.")

    def on_generate(self):
        # Action to perform when the button is clicked
        print("Generate button clicked!")



if __name__ == "__main__":
    '''
    This is an example of how to connect the widget and the signal it emits to make other changes to other widgets.
    '''

    app = QApplication(sys.argv)

    # Create a main window to embed the widget
    main_window = QWidget()
    main_window.setWindowTitle("Main Window with Image Processing App")
    main_window.setGeometry(100, 100, 1000, 600)

    # Create an instance of the image processing app
    image_processing_widget = BackgroundRemover()
    generate_widget = GenerateWidget()

    # Connect the signal to the slot to show the GenerateWidget and pass images
    image_processing_widget.update_complete.connect(generate_widget.set_images)
    image_processing_widget.update_complete.connect(generate_widget.show)

    # Set the layout for the main window
    main_layout = QVBoxLayout(main_window)
    main_layout.addWidget(image_processing_widget)
    main_layout.addWidget(generate_widget)
    
    # Hide the generate widget at first
    generate_widget.hide()
    # Set layout
    main_window.setLayout(main_layout)
    main_window.show()

    sys.exit(app.exec_())
