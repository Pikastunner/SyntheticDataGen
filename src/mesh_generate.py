'''
This module handles the creation of the mesh object (.obj)

'''


import sys
import numpy as np
import cv2
import cv2.aruco as aruco
from PyQt5.QtWidgets import (QApplication, QWidget, QPushButton, QVBoxLayout)

from image_segment import BackgroundRemover

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

import open3d as o3d

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


    def generate_mesh(self, rgb_image, depth_image):
        # Detect ArUco markers in the RGB image
        gray_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2GRAY)
        dictionary = aruco.getPredefinedDictionary(aruco.DICT_4X4_250)
        parameters = aruco.DetectorParameters()
        detector = aruco.ArucoDetector(dictionary, parameters)
        corners, ids, _ = detector.detectMarkers(gray_image)

        # store in current directory as "obj.obj and open"

        # If markers detected, you can process them further
        if ids is not None:
            print(f"Detected ArUco markers: {ids}")
        
        # Generate point cloud from RGB and depth images
        h, w = depth_image.shape
        fx, fy = 525.0, 525.0  # Example focal lengths, adjust as necessary
        cx, cy = w / 2, h / 2
        
        # Create an empty point cloud
        points = []
        colors = []
        
        for v in range(h):
            for u in range(w):
                Z = depth_image[v, u] / 1000.0  # Convert depth to meters
                if Z == 0:  # Skip points with no depth info
                    continue
                X = (u - cx) * Z / fx
                Y = (v - cy) * Z / fy
                points.append([X, Y, Z])
                colors.append(rgb_image[v, u] / 255.0)  # Normalize color
        
        # Convert points and colors to numpy arrays
        points = np.array(points)
        colors = np.array(colors)
        
        # Create an Open3D PointCloud object
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors)
        
        # Estimate normals for the point cloud
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
        
        # Create mesh using Poisson surface reconstruction
        mesh, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=9)
        
        # Save the mesh as an .obj file
        o3d.io.write_triangle_mesh("object_mesh.obj", mesh)
        
        # Optional: Visualize the mesh
        o3d.visualization.draw_geometries([mesh])


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
