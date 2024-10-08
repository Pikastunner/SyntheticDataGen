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

# class MeshGenerator(QWidget):
#     def __init__(self):
#         super().__init__()

#         # Create layout
#         self.layout = QVBoxLayout(self)

#         # Create a button with "Generate" label
#         self.generate_button = QPushButton("Generate", self)
#         self.generate_button.clicked.connect(self.on_generate)  # Connect button to the slot method
#         self.layout.addWidget(self.generate_button)

#         # Canvas for displaying the plot
#         self.canvas = FigureCanvas(Figure())
#         self.layout.addWidget(self.canvas)

#         # Variables to hold images
#         self.extracted_images = []
#         self.depth_images = []
#         self.aruco_images = []

#     def set_images(self, extracted_objects, extracted_depths, extracted_arucos):
#         # Store the images received from BackgroundRemover
#         self.extracted_images = extracted_objects
#         self.depth_images = extracted_depths
#         self.aruco_images = extracted_arucos
#         print(f"Received {len(extracted_objects)} Extracted images, {len(extracted_depths)} Depth images and {len(extracted_arucos)} Aruco images.")

#     def on_generate(self):
#         print("Generate button clicked!")

#         if not self.extracted_images or not self.depth_images:
#             print("No images or depth data available.")
#             return

#         ## TODO Calculate point cloud of each extracted object with image of its depth with depth info only where aruco and object are, the image with just the object and the image with just the aruco markers


#     def generate_mesh(self, rgb_image, depth_image):
#         # Example of how to use cv2.aruco library
#         gray_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2GRAY)
#         dictionary = aruco.getPredefinedDictionary(aruco.DICT_4X4_250)
#         parameters = aruco.DetectorParameters()
#         detector = aruco.ArucoDetector(dictionary, parameters)
#         corners, ids, _ = detector.detectMarkers(gray_image)

#         # If markers detected, you can process them further
#         if ids is not None:
#             print(f"Detected ArUco markers: {ids}")
    


# if __name__ == "__main__":
#     '''
#     This is an example of how to connect the widget and the signal it emits to make other changes to other widgets.
#     '''

#     app = QApplication(sys.argv)

#     # Create a main window to embed the widget
#     main_window = QWidget()
#     main_window.setWindowTitle("Mesh Generator")
#     main_window.setGeometry(100, 100, 1000, 600)

#     # Create an instance of the image processing app
#     image_processing_widget = BackgroundRemover()
#     generate_widget = MeshGenerator()

#     # Connect the signal to the slot to show the GenerateWidget and pass images
#     image_processing_widget.update_complete.connect(generate_widget.set_images)
#     image_processing_widget.update_complete.connect(generate_widget.show)

#     # Set the layout for the main window
#     main_layout = QVBoxLayout(main_window)
#     main_layout.addWidget(image_processing_widget)
#     main_layout.addWidget(generate_widget)
    
#     # Hide the generate widget at first
#     generate_widget.hide()
#     # Set layout
#     main_window.setLayout(main_layout)
#     main_window.show()

#     sys.exit(app.exec_())



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
        self.aruco_images = []

        # Use default intrinsic parameters for the D435i camera
        self.camera_matrix = np.array([[615.20, 0, 320.00],
                                       [0, 615.20, 240.00],
                                       [0, 0, 1]])
        self.dist_coeffs = np.zeros(5)  # Assuming no distortion


    def set_images(self, extracted_objects, extracted_depths, extracted_arucos):
        # Store the images received from BackgroundRemover
        self.extracted_images = extracted_objects
        self.depth_images = extracted_depths
        self.aruco_images = extracted_arucos
        print(f"Received {len(extracted_objects)} Extracted images, {len(extracted_depths)} Depth images and {len(extracted_arucos)} Aruco images.")

    def on_generate(self):
        print("Generate button clicked!")

        if not self.extracted_images or not self.depth_images:
            print("No images or depth data available.")
            return

        combined_point_cloud = o3d.geometry.PointCloud()

        for rgb_image, depth_image, aruco_image in zip(self.extracted_images, self.depth_images, self.aruco_images):
            point_cloud, pose_matrix = self.generate_point_cloud_with_pose(rgb_image, depth_image, aruco_image)

            if point_cloud is None:
                print("No valid point cloud generated from this image.")
                continue

            # Apply pose transformation
            transformed_point_cloud = point_cloud.transform(pose_matrix)

            # Combine point clouds
            combined_point_cloud += transformed_point_cloud

        # Compute normals for the combined point cloud
        combined_point_cloud.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))

        # Downsample the point cloud
        combined_point_cloud = combined_point_cloud.voxel_down_sample(voxel_size=0.01)

        # Generate mesh using Poisson surface reconstruction
        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(combined_point_cloud, depth=9)

        # Visualize the mesh
        o3d.visualization.draw_geometries([mesh])

    
    def generate_point_cloud_with_pose(self, rgb_image, depth_image, aruco_image):
        """
        Generates a point cloud from the depth image and calculates the camera pose using ArUco markers.
        """
        # Detect ArUco markers in the ArUco image
        gray_aruco_image = cv2.cvtColor(aruco_image, cv2.COLOR_BGR2GRAY)
        dictionary = aruco.getPredefinedDictionary(aruco.DICT_4X4_250)
        parameters = aruco.DetectorParameters()
        detector = aruco.ArucoDetector(dictionary, parameters)
        corners, ids, _ = detector.detectMarkers(gray_aruco_image)

        if ids is None:
            print("No ArUco markers detected in this image.")
            return None, None

        # Use solvePnP for each detected marker
        object_points = np.array([[0, 0, 0], [0, 0.05, 0], [0.05, 0.05, 0], [0.05, 0, 0]], dtype=np.float32)  # Example for a square marker
        pose_matrices = []
        for i in range(len(ids)):
            image_points = corners[i][0]  # Get the corners of the detected marker
            retval, rvec, tvec = cv2.solvePnP(object_points, image_points, self.camera_matrix, self.dist_coeffs)
            if retval:
                # Convert rotation vector to rotation matrix
                rotation_matrix, _ = cv2.Rodrigues(rvec)
                pose_matrix = np.eye(4)
                pose_matrix[:3, :3] = rotation_matrix
                pose_matrix[:3, 3] = tvec.flatten()
                pose_matrices.append(pose_matrix)

        # Use the first pose matrix (or combine them as needed)
        pose_matrix = pose_matrices[0] if pose_matrices else np.eye(4)

        # Convert depth image to point cloud using Open3D
        depth_o3d = o3d.geometry.Image(depth_image)
        rgb_o3d = o3d.geometry.Image(rgb_image)

        rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(rgb_o3d, depth_o3d)
        intrinsic = o3d.camera.PinholeCameraIntrinsic(rgb_image.shape[1], rgb_image.shape[0], 
                                                    self.camera_matrix[0, 0], self.camera_matrix[1, 1],
                                                    self.camera_matrix[0, 2], self.camera_matrix[1, 2])
        
        point_cloud = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, intrinsic)
        
        return point_cloud, pose_matrix



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

