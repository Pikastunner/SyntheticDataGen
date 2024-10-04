'''
This module handles the creation of the mesh object (.obj)

'''


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

from image_segment import BackgroundRemover


from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

import matplotlib as plt
from matplotlib import cm


class MeshGenerator(QWidget):
    def __init__(self):
        super().__init__()

        # Create layout
        self.layout = QVBoxLayout(self)

        # Create a button with "Generate" label
        self.generate_button = QPushButton("Generate", self)
        self.generate_button.clicked.connect(self.on_generate)  # Connect button to the slot method
        self.layout.addWidget(self.generate_button)

        # Canvas for displaying the plot
        self.canvas = FigureCanvas(Figure())
        self.layout.addWidget(self.canvas)

        # Variables to hold images
        self.extracted_images = []
        self.depth_images = []

    def set_images(self, extracted_images, depth_images):
        # Store the images received from BackgroundRemover
        self.extracted_images = extracted_images
        self.depth_images = depth_images
        print(f"Received {len(extracted_images)} Extracted images and {len(depth_images)} Depth images.")

    def on_generate(self):
        print("Generate button clicked!")

        if not self.extracted_images or not self.depth_images:
            print("No images or depth data available.")
            return

        rgb_image = self.extracted_images[0]
        depth_image = self.depth_images[0]

        # Generate the mesh using the depth image
        self.generate_mesh(rgb_image, depth_image)


    # def generate_mesh(self, rgb_image, depth_image):
    #     # Resize the RGB image to match the depth image size
    #     height, width = depth_image.shape
    #     if rgb_image.shape[:2] != (height, width):
    #         rgb_image_resized = cv2.resize(rgb_image, (width, height))
    #     else:
    #         rgb_image_resized = rgb_image

    #     # Generate the mesh grid
    #     x = np.arange(0, width)
    #     y = np.arange(0, height)
    #     X, Y = np.meshgrid(x, y)
    #     Z = depth_image

    #     # Normalize the RGB values to the range [0, 1]
    #     rgb_norm = rgb_image_resized / 255.0

    #     # Get the figure and axis from the canvas
    #     ax = self.canvas.figure.add_subplot(111, projection='3d')

    #     # Clear the axis for new plot
    #     ax.clear()

    #     # Plot the 3D surface mesh using RGB as face colors
    #     ax.plot_surface(X, Y, Z, facecolors=rgb_norm, rstride=1, cstride=1, antialiased=False)

    #     # Refresh the canvas to update the display
    #     self.canvas.draw()
    def generate_mesh(self, rgb_image, depth_image):
        # Resize the RGB image to match the depth image size
        height, width = depth_image.shape
        if rgb_image.shape[:2] != (height, width):
            rgb_image_resized = cv2.resize(rgb_image, (width, height))
        else:
            rgb_image_resized = rgb_image

        # Detect ArUco markers in the RGB image
        aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)  # Adjust the dictionary based on your setup
        aruco_params = aruco.DetectorParameters_create()
        gray_image = cv2.cvtColor(rgb_image_resized, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = aruco.detectMarkers(gray_image, aruco_dict, parameters=aruco_params)

        if ids is not None:
            print(f"Detected {len(ids)} ArUco marker(s) with IDs: {ids.flatten()}")

            # Iterate through each detected marker
            for i, marker_corners in enumerate(corners):
                # Get the marker's corners and ID
                marker_corners = marker_corners.reshape((4, 2))
                top_left, top_right, bottom_right, bottom_left = marker_corners
                marker_id = ids[i][0]

                # Optionally draw the detected marker
                cv2.polylines(rgb_image_resized, [np.int32(marker_corners)], isClosed=True, color=(0, 255, 0), thickness=2)
                cv2.putText(rgb_image_resized, f'ID: {marker_id}', (int(top_left[0]), int(top_left[1]) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                # Example: Use marker's top-left corner as a reference for mesh alignment
                # You can use the corners to adjust or scale your depth map, mesh, or RGB image

        else:
            print("No ArUco markers detected.")

        # Continue with generating the mesh grid
        x = np.arange(0, width)
        y = np.arange(0, height)
        X, Y = np.meshgrid(x, y)
        Z = depth_image

        # Normalize the RGB values to the range [0, 1]
        rgb_norm = rgb_image_resized / 255.0

        # Get the figure and axis from the canvas
        ax = self.canvas.figure.add_subplot(111, projection='3d')

        # Clear the axis for new plot
        ax.clear()

        # Plot the 3D surface mesh using RGB as face colors
        ax.plot_surface(X, Y, Z, facecolors=rgb_norm, rstride=1, cstride=1, antialiased=False)

        # Refresh the canvas to update the display
        self.canvas.draw()


if __name__ == "__main__":
    '''
    This is an example of how to connect the widget and the signal it emits to make other changes to other widgets.
    '''

    app = QApplication(sys.argv)

    # Create a main window to embed the widget
    main_window = QWidget()
    main_window.setWindowTitle("Mesh Generator")
    main_window.setGeometry(100, 100, 1000, 600)

    # Create an instance of the image processing app
    image_processing_widget = BackgroundRemover()
    generate_widget = MeshGenerator()

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
