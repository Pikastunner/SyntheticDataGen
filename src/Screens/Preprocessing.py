import os
import cv2
import numpy as np
from PyQt5.QtWidgets import (QWidget, QLineEdit, QPushButton, QLabel, QVBoxLayout, QHBoxLayout, 
                             QFileDialog, QMessageBox, QSizePolicy)
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import Qt
import cv2.aruco as aruco
from rembg import remove
from PIL import Image

from params import *

import concurrent.futures

import open3d as o3d

OUTPUT_PATH = "input_images"

import concurrent.futures
import time
from scipy.spatial import Delaunay  # Ensure this import is included



# Preprocessing Page
class PreprocessingScreen(QWidget):

    def update_variables(self, rgb_filenames, depth_filenames):
        self.processed_images, self.depth_images, self.aruco_datas = self.process_images(rgb_filenames, depth_filenames)
        self.image_index = 0

        ## PROCESS ALL IMAGES WITH ANNOTATION OF THE CENTRE OF THE ARUCO MARKERS
        self.annotated_images = []
        # Annotate image with board centre
        for i, img in enumerate(self.processed_images):
            img = self.processed_images[i]

            corners = self.aruco_datas[i][0]
            ids = self.aruco_datas[i][1]

            if len(ids) < 2:
                continue
            objpoints, imgpoints = aruco_board().matchImagePoints(corners, ids)

            if objpoints is None or imgpoints is None or not objpoints.any() or not imgpoints.any():
                continue
            if len(objpoints) < 4 or len(imgpoints) < 4:
                continue

            _, rvec, tvec = cv2.solvePnP(objectPoints=objpoints, imagePoints=imgpoints, cameraMatrix=camera_matrix(), distCoeffs=dist_coeffs(), flags=cv2.SOLVEPNP_ITERATIVE)
            self.annotated_images.append(cv2.drawFrameAxes(img.copy(), cameraMatrix=camera_matrix(), distCoeffs=dist_coeffs(), rvec=rvec, tvec=tvec , thickness=3, length=0.02))
            
            
        ## DISPLAY THE FIRST IMAGE
        qimage = self.numpy_to_qimage(self.annotated_images[self.image_index])
        self.background_image.setPixmap(QPixmap.fromImage(qimage))
        self.background_image.setScaledContents(True)  # Allow the pixmap to scale with the label
        self.background_image.setGeometry(self.rect())  # Make QLabel cover the whole widget
        self.background_image.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)  # Set size policy
        self.background_image_info.setText(f"Image #1 of {len(self.processed_images)}")

        ## GET POINT CLOUD
        self.accumulated_point_cloud = self.generate_point_cloud()

        self.triangle_mesh = PreprocessingScreen.generate_mesh_from_pcl(self.accumulated_point_cloud)
        self.graphical_interface_image.setPixmap(self.point_cloud_to_image(self.accumulated_point_cloud))

    ############################################################
            # CREATE/PROCESS A MESH
    ############################################################

    @staticmethod
    def generate_mesh_from_pcl(pcl, alpha=0.1):
        """
        Generate a mesh from a point cloud using alpha shapes.

        Parameters:
        pcl (o3d.geometry.PointCloud or numpy.ndarray): A Nx3 array of points (and optionally colors).
        alpha (float): Alpha parameter for the alpha shape algorithm.

        Returns:
        open3d.geometry.TriangleMesh: The reconstructed mesh.
        """
        pcl_np = np.asarray(pcl.points)
        colors_np = np.asarray(pcl.colors)  # Get colors as well
       
        tri = Delaunay(pcl_np[:, :2])  # Use only x and y for triangulation

        # Calculate the circumradius of each triangle
        triangles = pcl_np[tri.simplices]
        a = np.linalg.norm(triangles[:, 1] - triangles[:, 0], axis=1)
        b = np.linalg.norm(triangles[:, 2] - triangles[:, 1], axis=1)
        c = np.linalg.norm(triangles[:, 2] - triangles[:, 0], axis=1)
        s = (a + b + c) / 2  # Semi-perimeter
        area = np.sqrt(s * (s - a) * (s - b) * (s - c))  # Area of the triangle
        circumradius = (a * b * c) / (4 * area + 1e-10)  # Avoid division by zero

        # Filter triangles based on the circumradius and alpha value
        mask = circumradius < (1 / alpha)
        filtered_triangles = tri.simplices[mask]

        # Create a mesh 
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(pcl_np)
        mesh.triangles = o3d.utility.Vector3iVector(filtered_triangles)

        # Ensure that only the vertices in the filtered triangles are colored
        vertex_colors = np.zeros_like(pcl_np)  # Initialize an array for vertex colors
        vertex_colors[:len(colors_np)] = colors_np  # Assign colors from the point cloud

        # Assign colors to the vertices of the filtered triangles
        mesh.vertex_colors = o3d.utility.Vector3dVector(vertex_colors)

        # point_cloud = mesh.sample_points_poisson_disk(number_of_points=10000)
        # mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(point_cloud)[0]

        mesh.compute_vertex_normals()

        return mesh


    

    ############################################################
            # GUI BEHAVIOUR/DISPLAY
    ############################################################
    def __init__(self, parent):
        super().__init__()
        self.parent = parent

        self.accumulated_point_cloud = o3d.geometry.PointCloud()
        self.triangle_mesh = o3d.geometry.TriangleMesh()

        # Title Section
        title_area = QWidget(objectName="PreprocessingTitleArea")
        title_layout = QVBoxLayout(title_area)
        title_layout.addWidget(QLabel("Preprocessing", objectName="PreprocessingLabel"))

        # Preprocessing Results Section
        preprocessing_results_area = QWidget(objectName="PreprocessingResultsArea")
        preprocessing_area_layout = QHBoxLayout(preprocessing_results_area)
        preprocessing_area_layout.setContentsMargins(0, 0, 0, 0)
        preprocessing_area_layout.setSpacing(0)

        # Background Section
        background_section = QWidget(objectName="PreprocessingBackgroundSection")
        background_layout = QVBoxLayout(background_section)
        background_layout.addWidget(QLabel("Review removed background", objectName="PreprocessingTitle"), 10)

        self.background_image = QLabel()
        self.processed_images = []
        self.annotated_images = []
        self.background_image_info = QLabel(f"Image #1 of {len(self.processed_images)}", objectName="PreprocessingBackgroundImageInfo")
        self.background_image_info.setAlignment(Qt.AlignCenter)

        next_button = QPushButton("Next")
        next_button.setFixedSize(120, 40)
        next_button.clicked.connect(self.move_to_next)

        center_widget = QWidget()
        center_layout = QVBoxLayout(center_widget)
        center_layout.addWidget(self.background_image_info, alignment=Qt.AlignHCenter)
        center_layout.addWidget(next_button, alignment=Qt.AlignHCenter)

        background_layout.addWidget(self.background_image, 86)
        background_layout.addWidget(center_widget);

        # Graphical Interface Section
        graphical_interface_section = QWidget(objectName="PreprocessingGraphInterfaceSection")
        graphical_interface_layout = QVBoxLayout(graphical_interface_section)
        graphical_interface_layout.addWidget(QLabel("Graphical 3D interface of input image", objectName="PreprocessingGraphicalInterface"), 10)

        self.graphical_interface_image = QLabel(objectName="PreprocessingGraphicalInterfaceImage")
        pixmap = QPixmap(f"{OUTPUT_PATH}/rgb_image_1.png")
        self.graphical_interface_image.setPixmap(pixmap)
        self.graphical_interface_image.setScaledContents(True)
        self.graphical_interface_image.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        fs_button = QPushButton("View fullscreen")
        fs_button.setFixedSize(150, 40)
        fs_button.clicked.connect(self.view_3d_interface)

        graphical_interface_layout.addWidget(self.graphical_interface_image, 86)
        graphical_interface_layout.addWidget(fs_button, 4, alignment=Qt.AlignHCenter)

        preprocessing_area_layout.addWidget(background_section, 55)
        preprocessing_area_layout.addWidget(graphical_interface_section, 45)

        # Directory Saving Section
        directory_saving_area = QWidget(objectName="PreprocessingDirectoryArea")
        directory_saving_area.setContentsMargins(15, 15, 15, 15)
        directory_text_layout = QVBoxLayout(directory_saving_area)
        directory_text_layout.addWidget(QLabel("Select a directory to save the synthetic data.", objectName="PreprocessingDirectoryInstructions"))

        self.directory_input = QLineEdit(objectName="PreprocessingDirectoryInput")
        self.directory_input.setFixedHeight(25)

        browse_button = QPushButton("Browse")
        browse_button.setFixedHeight(25)
        browse_button.clicked.connect(self.select_directory)

        directory_input_layout = QHBoxLayout()
        directory_input_layout.addWidget(self.directory_input)
        directory_input_layout.addWidget(browse_button)
        directory_text_layout.addLayout(directory_input_layout)

        # Navigation Section
        navigation_area = QWidget(objectName="PreprocessingNavigationArea")
        navigation_buttons_layout = QHBoxLayout(navigation_area)
        spacer = QWidget()
        spacer.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)

        back_button = QPushButton("Back", objectName="BackPageButton")
        back_button.setFixedSize(100, 30)
        back_button.clicked.connect(self.go_to_back_page)

        next_button = QPushButton("Next", objectName="NextPageButton")
        next_button.setFixedSize(100, 30)
        next_button.clicked.connect(self.go_to_next_page)

        navigation_buttons_layout.addWidget(spacer)
        navigation_buttons_layout.addWidget(back_button)
        navigation_buttons_layout.addSpacing(10)
        navigation_buttons_layout.addWidget(next_button)
        navigation_buttons_layout.setAlignment(Qt.AlignRight | Qt.AlignBottom)

        # Main Layout
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        main_layout.addWidget(title_area, 10)
        main_layout.addWidget(preprocessing_results_area, 63)
        main_layout.addWidget(directory_saving_area, 17)
        main_layout.addWidget(navigation_area, 10)

    
    def view_3d_interface(self):
        # coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
        # o3d.visualization.draw_geometries([self.accumulated_point_cloud, coordinate_frame])
        coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
        o3d.visualization.draw_geometries([self.triangle_mesh, coordinate_frame])

    
    def select_directory(self):
        directory = QFileDialog.getExistingDirectory(self, "Select Directory")
        if directory:
            self.directory_input.setText(directory)

    def move_to_next(self):
        # Increment the index and wrap around if it exceeds the number of processed images
        self.image_index = (self.image_index + 1) % len(self.processed_images)
        self.display_current_image()  # Call to update the image display
        self.update_image_info()  # Update the label text

    def update_image_info(self):
        # Update the QLabel with the current image index
        self.background_image_info.setText(f"Image #{self.image_index + 1} of {len(self.processed_images)}")
    
    def display_current_image(self):
        if self.processed_images:  # Ensure there are images to display
            current_image = self.annotated_images[self.image_index]
        
            qimage = self.numpy_to_qimage(current_image)

            self.background_image.setPixmap(QPixmap.fromImage(qimage))
            self.background_image.setScaledContents(True)  # Optional: Scale to fit the label
            self.background_image.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)  # Set size policy


    ############################################################
            # LOAD/PROCESS DEPTH/RGB/ARUOCO
    ############################################################

    # Function to create a mask using rembg
    @staticmethod
    def create_mask_with_rembg(rgb_image):
        # Convert RGB image to grayscale for ArUco detection
        gray_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)

        # Define Aruco Parameters
        dictionary = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
        parameters = aruco.DetectorParameters()
        parameters.adaptiveThreshWinSizeMax = 80  # Increase max value for better handling of varying light
        parameters.errorCorrectionRate = 1
        # parameters.useAruco3Detection = True
        parameters.cornerRefinementMethod = aruco.CORNER_REFINE_SUBPIX
        parameters.cornerRefinementMaxIterations = 40

        # Detect ArUco markers before background removal
        detector = aruco.ArucoDetector(dictionary, parameters)
        corners, ids, rejected = detector.detectMarkers(gray_image)

        if ids is not None:
            corners, ids, _, recovered = detector.refineDetectedMarkers(gray_image, board=aruco_board(), detectedCorners=corners, detectedIds=ids, rejectedCorners=rejected,cameraMatrix=camera_matrix(),distCoeffs=dist_coeffs())
            print(f"{len(ids)} aruco markers found")
            print(ids)
        else:
            return None, None, (None, None)

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
        result_np = np.array(result_image)
        object_mask = result_np[:, :, 3]

        # Apply the inverted dilated mask to the object mask to exclude the markers and their surroundings
        inverted_dilated_aruco_mask = cv2.bitwise_not(dilated_aruco_mask)
        refined_object_mask = cv2.bitwise_and(object_mask, inverted_dilated_aruco_mask)

        # Return the refined object mask, the aruco mask and the aruco information
        return refined_object_mask, aruco_mask, (corners, ids)
    
    @staticmethod
    def apply_mask(rgb_image, mask):
        mask_3channel = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
        object_extracted = cv2.bitwise_and(rgb_image, mask_3channel)
        return object_extracted
    
    def load_rgb_images(self, rgb_filenames=None):
        folder_path = './input_images/' if not rgb_filenames else ''
        rgb_images = [cv2.cvtColor(cv2.imread(f"{folder_path}{filename}"), cv2.COLOR_BGR2RGB) for filename in rgb_filenames]
        return rgb_images

    def load_depth_images(self, depth_filenames=None):
        folder_path = './input_images/' if not depth_filenames else ''
        depth_images = [cv2.normalize(cv2.imread(f"{folder_path}{filename}", cv2.IMREAD_UNCHANGED), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8) for filename in depth_filenames]
        return depth_images


    @staticmethod
    def process_image(rgb_image, depth_image):
        mask, _, aruco_data = PreprocessingScreen.create_mask_with_rembg(rgb_image)
        if aruco_data == (None, None):
            return None, None, None
        elif len(aruco_data[1]) < 2:
            return None, None, None


        object_extracted = PreprocessingScreen.apply_mask(rgb_image, mask)
        return object_extracted, depth_image, aruco_data

    def process_images(self, rgb_filenames, depth_filenames):
        processed_images = []
        rgb_images = self.load_rgb_images(rgb_filenames)
        depth_images = self.load_depth_images(depth_filenames)
        aruco_returned = []
        depth_returned = []

        with concurrent.futures.ProcessPoolExecutor() as executor:
            futures = {executor.submit(self.process_image, rgb_images[i], depth_images[i]): i 
                       for i in range(min(len(rgb_images), len(depth_images)))}
            
            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                if result[0] is not None:
                    processed_images.append(result[0])
                    depth_returned.append(result[1])
                    aruco_returned.append(result[2])

        return processed_images, depth_returned, aruco_returned

    
    def numpy_to_qimage(self, img):
        height, width, channel = img.shape
        bytes_per_line = channel * width
        qimage = QImage(img.data, width, height, bytes_per_line, QImage.Format_RGB888)
        return qimage

    ############################################################
            # CREATE/PROCESS A POINT CLOUD
    ############################################################

    def point_cloud_to_image(self, point_cloud, image_size=(500, 500)):
        # Convert point cloud data to numpy arrays
        points = np.asarray(point_cloud.points)
        colors = np.asarray(point_cloud.colors) * 255  # Scale colors to 0-255 range if they are normalized

        # Create a blank image
        img = np.zeros((image_size[1], image_size[0], 3), dtype=np.uint8)
        
        # Extract x and y coordinates (z is ignored for 2D projection)
        x_coords = points[:, 1]
        y_coords = points[:, 2]

        # Normalize x and y coordinates to fit within image dimensions
        x_normalized = ((x_coords - x_coords.min()) / (x_coords.max() - x_coords.min()) * (image_size[0] - 1)).astype(int)
        y_normalized = ((y_coords - y_coords.min()) / (y_coords.max() - y_coords.min()) * (image_size[1] - 1)).astype(int)
        
        # Populate the image with colored points
        for x, y, color in zip(x_normalized, y_normalized, colors):
            img[y, x] = color.astype(np.uint8)  # Set each point color


        # Apply Gaussian blur to the image
        img = cv2.GaussianBlur(img, (15, 15), 0)  # Adjust kernel size as needed
        
        # Convert the image to a QImage
        height, width, channel = img.shape
        bytes_per_line = 3 * width
        qimage = QImage(img.data, width, height, bytes_per_line, QImage.Format_RGB888)

        return QPixmap.fromImage(qimage)
    
    @staticmethod
    def remove_scraggly_bits(point_cloud, eps=0.003, min_points=10):
        labels = np.array(point_cloud.cluster_dbscan(eps=eps, min_points=min_points, print_progress=False))
        largest_cluster_label = np.bincount(labels[labels >= 0]).argmax()
        
        return point_cloud.select_by_index(np.where(labels == largest_cluster_label)[0])


    @staticmethod
    def depth_to_point_cloud(rgb_image, depth_image):
        # Get camera matrix information
        start = time.time()
        matrix = camera_matrix_rgb() # TODO find out why for whatever reason this works bests for point cloud estimation
        
        h, w, channels = rgb_image.shape
        fx, fy = matrix[0, 0], matrix[1, 1]
        cx, cy = matrix[0, 2], matrix[1, 2]

        # Prepare the meshgrid of pixel coordinates
        u, v = np.meshgrid(np.arange(w), np.arange(h))
        
        # Get depth values and convert to meters
        Z = depth_image / 1000.0  # Convert depth to meters

        # Create masks for valid points
        valid_mask = (Z > 0) & (channels == 3) & (rgb_image[:, :, :3].any(axis=-1))

        # Compute the 3D points using valid masks
        X = ((u[valid_mask] - cx) * Z[valid_mask] / fx)
        Y = ((v[valid_mask] - cy) * Z[valid_mask] / fy)

        # Create point cloud data
        points = np.vstack((X, Y, Z[valid_mask])).T
        colors = rgb_image[valid_mask, :3] / 255.0  # Normalize RGB

        # Create Open3D point cloud
        point_cloud = o3d.geometry.PointCloud()
        point_cloud.points = o3d.utility.Vector3dVector(points)
        point_cloud.colors = o3d.utility.Vector3dVector(colors)

        print(f"Time taken to generate point cloud: {time.time() - start:.2f} seconds")

        return point_cloud


    def generate_point_cloud(self):
        if not self.processed_images or not self.depth_images or not self.aruco_datas:
            print("No images or depth data available.")
            return

        # Initialize accumulated point cloud and reuse constants
        accumulated_point_cloud = o3d.geometry.PointCloud()
        dist_coeffs_cached = dist_coeffs()

        for i, (rgb_image, depth_image, aruco_data) in enumerate(zip(self.processed_images, self.depth_images, self.aruco_datas)):
            ids, corners = aruco_data[1], aruco_data[0]

            ## CHECK FOR UNUSABLE DATA
            if ids is None or len(ids) < 2: # Dont bother if there arent enough markers
                continue
            objpoints, imgpoints = aruco_board().matchImagePoints(corners, ids)
            if objpoints is None or imgpoints is None or not objpoints.any() or not imgpoints.any():
                continue
            if len(objpoints) < 4 or len(imgpoints) < 4:
                continue
           
            _, rvec, tvec = cv2.solvePnP(
                objectPoints=objpoints,
                imagePoints=imgpoints,
                cameraMatrix=camera_matrix(),
                distCoeffs=dist_coeffs_cached,
                flags=cv2.SOLVEPNP_ITERATIVE
            )

            R, _ = cv2.Rodrigues(rvec)

            point_cloud = PreprocessingScreen.depth_to_point_cloud(rgb_image, depth_image)
            point_cloud = PreprocessingScreen.remove_scraggly_bits(point_cloud, min_points=30)
            
            # Compute transformation matrix
            transformation_matrix = np.eye(4)
            transformation_matrix[:3, :3] = R
            transformation_matrix[:3, 3] = tvec.flatten()
            transformation_matrix_inverse = np.linalg.inv(transformation_matrix)

            # Apply transformation to all points at once
            points = np.asarray(point_cloud.points)  # Convert points to a NumPy array
            points_homogeneous = np.c_[points, np.ones(points.shape[0])]
            transformed_points = points_homogeneous @ transformation_matrix_inverse.T
            point_cloud.points = o3d.utility.Vector3dVector(transformed_points[:, :3])

            if i == 0:
                accumulated_point_cloud = point_cloud
            else:
                # TODO disabling ICP algorithm for now
                # Apply ICP to align the current point cloud with the accumulated point cloud
                icp_result = o3d.pipelines.registration.registration_icp(
                    point_cloud, 
                    accumulated_point_cloud, 
                    max_correspondence_distance=0.0013,  # Adjust based on scale
                    estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint()
                )

                # Transform the current point cloud using the ICP result
                point_cloud.transform(icp_result.transformation)
                accumulated_point_cloud += point_cloud         
        
        # Remove any outliers that arent connected to largest point cloud
        accumulated_point_cloud = PreprocessingScreen.remove_scraggly_bits(accumulated_point_cloud, min_points=4)       

        return accumulated_point_cloud
    




    ############################################################
            # OVERARCHING PAGE CONTROL
    ############################################################
        
    def go_to_back_page(self):
        current_index = self.parent.currentIndex()
        if current_index > 0:
            self.parent.setCurrentIndex(current_index - 1) 
        else:
            print("Already on the first page")

    def go_to_next_page(self):
        if (len(self.directory_input.text()) == 0):
            error_msg = QMessageBox()
            
            # Set critical icon for error message
            error_msg.setIcon(QMessageBox.Critical)
            
            # Set the title of the error message box
            error_msg.setWindowTitle("Empty Path Error")
            
            # Set the detailed text to help the user troubleshoot
            error_msg.setText('<span style="color:#005ace;font-size: 15px;">No Path Input!</span>')
            error_msg.setInformativeText("Please make sure you specify a path")
            
            # Set the standard button to close the message box
            error_msg.setStandardButtons(QMessageBox.Ok)

            # Execute and show the message box
            error_msg.exec_()
            return
        current_index = self.parent.currentIndex()
        if current_index < self.parent.count() - 1:
            self.parent.setCurrentIndex(current_index + 1)
        else:
            print("Already on the last page")
