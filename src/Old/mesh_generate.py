# # '''
# # This module handles the creation of the mesh object (.obj)

# # '''


import sys
import numpy as np
import cv2
import cv2.aruco as aruco
from PyQt5.QtWidgets import (QApplication, QWidget, QPushButton, QVBoxLayout)

from Old.image_segment import BackgroundRemover

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

import open3d as o3d

import numpy.linalg as la

from params import *

# Add necessary imports at the beginning
from PyQt5.QtWidgets import (QApplication, QWidget, QPushButton, QVBoxLayout, QLabel, QSlider)

class MeshGenerator(QWidget):
    def __init__(self):
        super().__init__()

        # Create layout
        self.layout = QVBoxLayout(self)

        # Create a button with "Generate" label
        self.generate_button = QPushButton("Generate", self)
        self.generate_button.clicked.connect(self.on_generate)
        self.layout.addWidget(self.generate_button)

        # Create a label to display the threshold value
        self.threshold_label = QLabel("Threshold: 0.04", self)
        self.layout.addWidget(self.threshold_label)

        # Create a slider for threshold adjustment
        self.threshold_slider = QSlider()
        self.threshold_slider.setOrientation(1)  # Horizontal orientation
        self.threshold_slider.setRange(1, 100)  # Set range from 1 to 100 (you can adjust this range)
        self.threshold_slider.setValue(4)  # Default value corresponds to 0.04
        self.threshold_slider.valueChanged.connect(self.update_threshold_label)
        self.layout.addWidget(self.threshold_slider)

        # Variables to hold images
        self.extracted_images = []
        self.depth_images = []
        self.aruco_data = []

        # Initialize threshold variable
        self.threshold = 0.04

    def update_threshold_label(self):
        # Update threshold value based on slider position
        self.threshold = self.threshold_slider.value() / 100.0  # Convert to float
        self.threshold_label.setText(f"Threshold: {self.threshold:.2f}")  # Update label display

    def set_data(self, extracted_objects, extracted_depths, extracted_arucos):
        # Store the images received from BackgroundRemover
        self.extracted_images = extracted_objects  # extracted objects with transparent background
        self.depth_images = extracted_depths  # depth information
        self.aruco_data = extracted_arucos  # list of (corners, ids) from aruco
        print(f"Received {len(extracted_objects)} Extracted images, {len(extracted_depths)} Depth images and {len(extracted_arucos)} Aruco data.")

    def depth_to_point_cloud(self, rgb_image, depth_image):
        """Convert depth map and RGB/RGBA image to point cloud, ignoring pure black pixels in both RGB and depth images."""
        matrix = camera_matrix()
        h, w, channels = rgb_image.shape
        fx, fy = matrix[0, 0], matrix[1, 1]
        cx, cy = matrix[0, 2], matrix[1, 2]

        # Check if the image has an alpha channel (RGBA)
        has_alpha = (channels == 4)

        # Create lists of 3D points and corresponding colors
        points = []
        colors = []

        for v in range(h):
            for u in range(w):
                Z = (depth_image[v, u] / 1000.0) # Convert depth to meters
                
                # Skip if depth is zero (black in the depth map)
                if Z == 0:
                    continue

                # Skip pixels that are transparent if alpha is present
                if has_alpha and rgb_image[v, u, 3] == 0:
                    continue

                # Skip pixels that are pure black in the RGB image
                rgb_color = rgb_image[v, u, :3]
                if np.all(rgb_color == 0):
                    continue

                # Compute the 3D point from depth and camera intrinsics
                X = ((u - cx) * Z / fx)
                Y = ((v - cy) * Z / fy)
                points.append([X, Y, Z])

                # Add the RGB color (ignoring the alpha channel if present)
                colors.append(rgb_color / 255.0)  # Normalize RGB

        # Create Open3D point cloud
        point_cloud = o3d.geometry.PointCloud()
        point_cloud.points = o3d.utility.Vector3dVector(np.array(points))
        point_cloud.colors = o3d.utility.Vector3dVector(np.array(colors))

        return point_cloud

    def remove_scraggly_bits(self, point_cloud, eps=0.003, min_points=10):
        labels = np.array(point_cloud.cluster_dbscan(eps=eps, min_points=min_points, print_progress=False))
        largest_cluster_label = np.bincount(labels[labels >= 0]).argmax()
        return point_cloud.select_by_index(np.where(labels == largest_cluster_label)[0])

    def on_generate(self):
        print("Generate button clicked!")
        if not self.extracted_images or not self.depth_images or not self.aruco_data:
            print("No images or depth data available.")
            return

        # Initialize accumulated point cloud and reuse constants
        accumulated_point_cloud = o3d.geometry.PointCloud()
        dist_coeffs_cached = dist_coeffs()
        
        for i, (rgb_image, depth_image, aruco_data) in enumerate(zip(self.extracted_images, self.depth_images, self.aruco_data)):
            ids, corners = aruco_data[1], aruco_data[0]
            objpoints, imgpoints = aruco_board().matchImagePoints(corners, ids)

            _, rvec, tvec = cv2.solvePnP(
                objectPoints=objpoints,
                imagePoints=imgpoints,
                cameraMatrix=camera_matrix(),
                distCoeffs=dist_coeffs_cached,
                flags=cv2.SOLVEPNP_ITERATIVE
            )

            R, _ = cv2.Rodrigues(rvec)

            point_cloud = self.depth_to_point_cloud(rgb_image, depth_image)
            point_cloud = self.remove_scraggly_bits(point_cloud, min_points=30)
            
            # Compute transformation matrix
            transformation_matrix = np.eye(4)
            transformation_matrix[:3, :3] = R
            transformation_matrix[:3, 3] = tvec.flatten()
            transformation_matrix_inverse = np.linalg.inv(transformation_matrix)

            # Apply transformation to all points at once
            points = np.asarray(point_cloud.points)  # Convert points to a NumPy array
            points_homogeneous = np.hstack((points, np.ones((points.shape[0], 1))))
            transformed_points = (transformation_matrix_inverse @ points_homogeneous.T).T[:, :3]

            point_cloud.points = o3d.utility.Vector3dVector(transformed_points)

            if i == 0:
                accumulated_point_cloud = point_cloud
            else:
                
                # Apply ICP to align the current point cloud with the accumulated point cloud
                icp_result = o3d.pipelines.registration.registration_icp(
                    point_cloud, 
                    accumulated_point_cloud, 
                    max_correspondence_distance=0.01,  # Adjust based on scale
                    estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint()
                )

                # Transform the current point cloud using the ICP result
                point_cloud.transform(icp_result.transformation)
                accumulated_point_cloud += point_cloud                

        # Visualize the accumulated point cloud
        accumulated_point_cloud.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.05, max_nn=30)  # Adjust radius and max_nn as needed
        )

        coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
        o3d.visualization.draw_geometries([accumulated_point_cloud, coordinate_frame])

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

    depth_files = []
    rgb_files = []

    for i in range(10,14):
        depth_files.append(f'input_images_2/depth_image_{i}.png')
        rgb_files.append(f'input_images_2/rgb_image_{i}.png')

    # Connect the signal to the slot to show the GenerateWidget and pass images
    image_processing_widget.update_complete.connect(generate_widget.set_data)
    image_processing_widget.update_complete.connect(generate_widget.show)

    image_processing_widget.set_rgbs(rgb_files)
    image_processing_widget.set_depths(depth_files)

    # Set the layout for the main window
    main_layout = QVBoxLayout(main_window)
    main_layout.addWidget(image_processing_widget)
    main_layout.addWidget(generate_widget)
    
    # Hide the generate widget at first
    # generate_widget.hide()
    # Set layout
    main_window.setLayout(main_layout)
    main_window.show()

    sys.exit(app.exec_())
