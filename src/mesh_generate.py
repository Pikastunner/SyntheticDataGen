# '''
# This module handles the creation of the mesh object (.obj)

# '''


import sys
import numpy as np
import cv2
import cv2.aruco as aruco
from PyQt5.QtWidgets import (QApplication, QWidget, QPushButton, QVBoxLayout)

from image_segment import BackgroundRemover

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

import open3d as o3d

import numpy.linalg as la


class MeshGenerator(QWidget):
    def __init__(self):
        super().__init__()

        # Create layout
        self.layout = QVBoxLayout(self)

        # Create a button with "Generate" label
        self.generate_button = QPushButton("Generate", self)
        self.generate_button.clicked.connect(self.on_generate) 
        self.layout.addWidget(self.generate_button)

        # Variables to hold images
        self.extracted_images = []
        self.depth_images = []
        self.aruco_data = []

        # These are the intrinsics of the D435i camera
        self.camera_matrix = np.array([[606.86, 0, 321.847],
                                       [0, 606.86, 244.995],
                                       [0, 0, 1]])
        self.dist_coeffs = np.zeros(5)  # Assuming no distortion


    def set_data(self, extracted_objects, extracted_depths, extracted_arucos):
        # Store the images received from BackgroundRemover
        self.extracted_images = extracted_objects # extracted objects with transparent background
        self.depth_images = extracted_depths  # depth information
        self.aruco_data = extracted_arucos  # list of (corners, ids) from aruco 
        print(f"Received {len(extracted_objects)} Extracted images, {len(extracted_depths)} Depth images and {len(extracted_arucos)} Aruco data.")

    def depth_to_point_cloud(self, rgb_image, depth_image):
        """Convert depth map and RGB/RGBA image to point cloud, ignoring pure black pixels in both RGB and depth images."""
        h, w, channels = rgb_image.shape
        fx, fy = self.camera_matrix[0, 0], self.camera_matrix[1, 1]
        cx, cy = self.camera_matrix[0, 2], self.camera_matrix[1, 2]

        # Check if the image has an alpha channel (RGBA)
        has_alpha = (channels == 4)

        # Create lists of 3D points and corresponding colors
        points = []
        colors = []

        for v in range(h):
            for u in range(w):
                Z = depth_image[v, u] / 1000.0  # Convert depth to meters
                
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
                X = (u - cx) * Z / fx
                Y = (v - cy) * Z / fy
                points.append([X, Y, Z])

                # Add the RGB color (ignoring the alpha channel if present)
                colors.append(rgb_color / 255.0)  # Normalize RGB

        # Create Open3D point cloud
        point_cloud = o3d.geometry.PointCloud()
        point_cloud.points = o3d.utility.Vector3dVector(np.array(points))
        point_cloud.colors = o3d.utility.Vector3dVector(np.array(colors))

        return point_cloud

    
    def remove_scraggly_bits(self, point_cloud, eps=0.003, min_points=10):
        """
        Remove scraggly bits by keeping only the largest cluster in the point cloud.
        
        Parameters:
        - point_cloud: The input Open3D point cloud object.
        - eps: The distance threshold for DBSCAN clustering.
        - min_points: The minimum number of points required to form a cluster.
        
        Returns:
        - A point cloud containing only the largest cluster.
        """
        # Perform DBSCAN clustering on the point cloud
        labels = np.array(point_cloud.cluster_dbscan(eps=eps, min_points=min_points, print_progress=True))

        # Find the largest cluster (ignoring noise points, labeled as -1)
        largest_cluster_label = np.bincount(labels[labels >= 0]).argmax()

        # Select points that belong to the largest cluster
        main_cluster = point_cloud.select_by_index(np.where(labels == largest_cluster_label)[0])

        return main_cluster


    # def on_generate(self):
    #     print("Generate button clicked!")

    #     if not self.extracted_images or not self.depth_images:
    #         print("No images or depth data available.")
    #         return

    #     # Initialize an empty point cloud to accumulate all clouds
    #     accumulated_point_cloud = o3d.geometry.PointCloud()

    #     for i, (rgb_image, depth_image, aruco_data) in enumerate(zip(self.extracted_images, self.depth_images, self.aruco_data)):
    #         # Convert depth map to point cloud
    #         point_cloud = self.depth_to_point_cloud(rgb_image, depth_image)

    #         point_cloud = self.remove_scraggly_bits(point_cloud, min_points=50)

    #         ids = aruco_data[1]
    #         corners = aruco_data[0]

    #         # Accumulate the transformed point clouds using ICP
    #         if i == 0:
    #             accumulated_point_cloud = point_cloud
    #         else:
    #             # Use the previous point cloud as a reference and perform ICP alignment
    #             threshold = 0.02  # Adjust the threshold depending on your data
    #             reg_p2p = o3d.pipelines.registration.registration_icp(
    #                 point_cloud, accumulated_point_cloud, threshold, np.eye(4),
    #                 o3d.pipelines.registration.TransformationEstimationPointToPoint()
    #             )

    #             # Apply the transformation from ICP
    #             point_cloud.transform(reg_p2p.transformation)

    #             # Merge the aligned point cloud
    #             accumulated_point_cloud += point_cloud

    #     # Visualize the combined point cloud
    #     o3d.visualization.draw_geometries([accumulated_point_cloud])




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
    image_processing_widget.update_complete.connect(generate_widget.set_data)
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

