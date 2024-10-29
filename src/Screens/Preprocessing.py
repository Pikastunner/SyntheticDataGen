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
import re

from params import *

import open3d as o3d

import numpy.linalg as la

OUTPUT_PATH = "input_images"

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

            board = aruco_board()
            objpoints, imgpoints = board.matchImagePoints(corners, ids)

            _, rvec, tvec = cv2.solvePnP(objectPoints=objpoints, imagePoints=imgpoints, cameraMatrix=camera_matrix(), distCoeffs=dist_coeffs(), flags=cv2.SOLVEPNP_ITERATIVE)
            self.annotated_images.append(cv2.drawFrameAxes(img.copy(), cameraMatrix=camera_matrix(), distCoeffs=dist_coeffs(), rvec=rvec, tvec=tvec , thickness=3, length=0.02))

        ## DISPLAY THE FIRST IMAGE
        qimage = self.numpy_to_qimage(self.annotated_images[self.image_index])
        self.background_image.setPixmap(QPixmap.fromImage(qimage))
        self.background_image.setScaledContents(True)  # Allow the pixmap to scale with the label
        self.background_image.setGeometry(self.rect())  # Make QLabel cover the whole widget
        self.background_image.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)  # Set size policy
        self.background_image_info.setText(f"Image #1 of {len(self.processed_images)}")

        self.accumulated_point_cloud = self.generate_point_cloud()
        self.graphical_interface_image.setPixmap(self.point_cloud_to_image(self.accumulated_point_cloud))

        self.triangle_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(self.accumulated_point_cloud, depth=8)[0]
        
    ############################################################
            # GUI BEHAVIOUR/DISPLAY
    ############################################################
    def __init__(self, parent):
        super().__init__()
        self.parent = parent

        self.accumulated_point_cloud = o3d.geometry.PointCloud()
        self.triangle_mesh = o3d.geometry.TriangleMesh()

        # Set up the initial container
        title_layout = QVBoxLayout()
        title_area = QWidget()
        # title_area.setObjectName("TitleArea")
        title_area.setStyleSheet("background-color: #d9d9d9;")

        # Working within the initial container
        title_text_layout = QVBoxLayout()
        label = QLabel("Preprocessing")
        # label.setObjectName("PreprocessingLabel")
        label.setStyleSheet("font-size: 18px; margin: 15px;")
        title_text_layout.addWidget(label)
        title_area.setLayout(title_text_layout)

        # Update changes to initial container
        title_layout.addWidget(title_area)

        # Set up the initial container
        preprocessing_results_layout = QVBoxLayout()
        preprocessing_results_area = QWidget()
        # preprocessing_results_area.setObjectName("Preprocessing_results_area")
        preprocessing_results_area.setStyleSheet("background-color: #d9d9d9;")
        

        # Working within the initial container
        preprocessing_area_layout = QHBoxLayout()
        preprocessing_area_layout.setContentsMargins(0, 0, 0, 0)
        preprocessing_area_layout.setSpacing(0)

        background_section = QWidget()
        background_section.setStyleSheet("margin: 15px")
        background_layout = QVBoxLayout()

        background_title = QLabel("Review removed background")
        # background_title.setObjectName("")
        background_title.setStyleSheet("font-size: 12px;")

        self.background_image = QLabel()
        self.processed_images = []
        self.annotated_images = []

        # Center and reduce spacing between background_image_info and background_image_next
        self.background_image_info = QLabel(f"Image #1 of {len(self.processed_images)}")
        self.background_image_info.setStyleSheet("font-size: 12px;")
        self.background_image_info.setAlignment(Qt.AlignCenter)  # Center the text

        background_image_next = QPushButton("Next")
        background_image_next.clicked.connect(self.move_to_next)
        background_image_next.setStyleSheet("background-color: #ededed")
        background_image_next.setFixedSize(120, 55)

        # Add a layout to group the info and button together
        center_widget = QWidget()
        center_layout = QVBoxLayout()

        center_layout.addWidget(self.background_image_info, alignment=Qt.AlignHCenter)
        center_layout.addWidget(background_image_next, alignment=Qt.AlignHCenter)
        center_layout.setContentsMargins(0, 0, 0, 0)
        center_layout.setSpacing(0)
        center_widget.setLayout(center_layout)

        # Add the widgets to the main layout
        background_layout.addWidget(background_title, 10)
        background_layout.addWidget(self.background_image, 86)
        background_layout.addWidget(center_widget)

        background_section.setLayout(background_layout)

        graphical_interface_section = QWidget()
        graphical_interface_section.setStyleSheet("margin: 15px")
        graphical_interface_layout = QVBoxLayout()
        graphical_interface_title = QLabel("Graphical 3D interface of input image")
        graphical_interface_title.setStyleSheet("font-size: 12px;")
        
        self.graphical_interface_image = QLabel()
        self.graphical_interface_image.setStyleSheet("background-color: black")
        pixmap = QPixmap(f"{OUTPUT_PATH}/rgb_image_1.png")
        self.graphical_interface_image.setPixmap(pixmap)
        self.graphical_interface_image.setScaledContents(True)  # Allow the pixmap to scale with the label
        self.graphical_interface_image.setGeometry(self.rect())  # Make QLabel cover the whole widget
        self.graphical_interface_image.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)  # Set size policy


        graphical_interface_fs = QPushButton("View fullscreen")
        graphical_interface_fs.setStyleSheet("background-color: #ededed")
        graphical_interface_fs.clicked.connect(self.view_3d_interface)

        graphical_interface_fs.setFixedSize(150, 55)
        graphical_interface_layout.addWidget(graphical_interface_title, 10)
        graphical_interface_layout.addWidget(self.graphical_interface_image, 86)
        graphical_interface_layout.addWidget(graphical_interface_fs, 4, alignment=Qt.AlignHCenter)
        
        graphical_interface_section.setLayout(graphical_interface_layout)

        preprocessing_area_layout.addWidget(background_section, 55)
        preprocessing_area_layout.addWidget(graphical_interface_section, 45)

        preprocessing_results_area.setLayout(preprocessing_area_layout)

        # Update changes to initial container
        preprocessing_results_layout.addWidget(preprocessing_results_area)
        
        # Set up the initial container
        directory_saving_layout = QVBoxLayout()
        directory_saving_area = QWidget()
        directory_saving_area.setStyleSheet("background-color: #d9d9d9;")
        directory_saving_area.setContentsMargins(15, 15, 15, 15)

        # Working within the initial container
        directory_text_layout = QVBoxLayout()
        directory_instructions = QLabel("Select a directory to save the synthetic data.")
        directory_instructions.setStyleSheet("font-size: 12px;")

        directory_text_layout.addWidget(directory_instructions)
        directory_saving_area.setLayout(directory_text_layout)

        # Create directory input box and browse button
        self.directory_input = QLineEdit()
        self.directory_input.setFixedHeight(25)  # Set the desired height
        self.directory_input.setStyleSheet("background-color: #ededed; border: none;")

        browse_button = QPushButton("Browse")
        browse_button.setStyleSheet("background-color: #ededed;")
        browse_button.setFixedHeight(25)  # Set the desired height
        browse_button.clicked.connect(self.select_directory)

        # Create layout for input box and button
        directory_input_layout = QHBoxLayout()
        directory_input_layout.addWidget(self.directory_input)
        directory_input_layout.addWidget(browse_button)
        directory_text_layout.addLayout(directory_input_layout)

        # Update changes to initial container
        directory_saving_layout.addWidget(directory_saving_area)

        navigation_layout = QHBoxLayout()
        navigation_area = QWidget()
        navigation_area.setStyleSheet("background-color: #d9d9d9;")

        navigation_buttons_layout = QHBoxLayout()

        # Spacer to shift the buttons to the right
        spacer = QWidget()
        spacer.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        navigation_buttons_layout.addWidget(spacer)

        back_button = QPushButton("Back")
        back_button.setFixedSize(100, 30)
        back_button.setStyleSheet("background-color: #ededed;")
        back_button.clicked.connect(self.go_to_back_page)

        next_button = QPushButton("Next")
        next_button.setFixedSize(100, 30)
        next_button.setStyleSheet("background-color: #ededed;")
        next_button.clicked.connect(self.go_to_next_page)

        # Add the buttons to the layout with a small gap between them
        navigation_buttons_layout.addWidget(back_button)
        navigation_buttons_layout.addSpacing(10)  # Set the gap between the buttons
        navigation_buttons_layout.addWidget(next_button)

        # Align buttons to the right and bottom
        navigation_buttons_layout.setAlignment(Qt.AlignRight | Qt.AlignBottom)

        navigation_area.setLayout(navigation_buttons_layout)

        navigation_layout.addWidget(navigation_area)
        
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        
        main_layout.addLayout(title_layout, 10)
        main_layout.addLayout(preprocessing_results_layout, 63)
        main_layout.addLayout(directory_saving_layout, 17)
        main_layout.addLayout(navigation_layout, 10)
        

        self.setLayout(main_layout)
    
    def view_3d_interface(self):
        # Open fullscreen 3D preview (placeholder)
        # QMessageBox.information(self, "3D Interface", "Viewing 3D interface.")
        coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
        o3d.visualization.draw_geometries([self.accumulated_point_cloud, coordinate_frame])

    
    def select_directory(self):
        directory = QFileDialog.getExistingDirectory(self, "Select Directory")
        if directory:
            self.directory_input.setText(directory)
            # QMessageBox.information(self, "Directory Selected", f"Data will be saved to {directory}")
    
    def go_to_complete(self):
        pass
        # generate_3d_mesh()  # Simulate 3D mesh generation
        # self.parent.setCurrentIndex(4)

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
    

    def get_files_starting_with(self, folder_path, prefix):
        files = []
        for file in os.listdir(folder_path):
            if re.search(rf"{prefix}_[0-9]", file):
            #file.startswith(prefix):
                files.append(os.path.join(folder_path, file))
        return files
    
    def apply_mask(self, rgb_image, mask):
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
    
    def process_images(self, rgb_filenames, depth_filenames):
        processed_images = []
        rgb_images = self.load_rgb_images(rgb_filenames)
        depth_images = self.load_depth_images(depth_filenames)
        aruco_datas = []
        for i in range(min(len(rgb_images), len(depth_images))):
            # Extract object and extract aruco information
            mask, _, aruco_data = self.create_mask_with_rembg(rgb_images[i])

            object_extracted = self.apply_mask(rgb_images[i], mask)
            processed_images.append(object_extracted)
            aruco_datas.append(aruco_data)
        return processed_images, depth_images, aruco_datas
    
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
        x_coords = points[:, 0]
        y_coords = points[:, 1]

        # Normalize x and y coordinates to fit within image dimensions
        x_normalized = ((x_coords - x_coords.min()) / (x_coords.max() - x_coords.min()) * (image_size[0] - 1)).astype(int)
        y_normalized = ((y_coords - y_coords.min()) / (y_coords.max() - y_coords.min()) * (image_size[1] - 1)).astype(int)
        
        # Populate the image with colored points
        for x, y, color in zip(x_normalized, y_normalized, colors):
            img[y, x] = color.astype(np.uint8)  # Set each point color
        
        # Convert the image to a QImage
        height, width, channel = img.shape
        bytes_per_line = 3 * width
        qimage = QImage(img.data, width, height, bytes_per_line, QImage.Format_RGB888)
        
        # Convert QImage to QPixmap
        qpixmap = QPixmap.fromImage(qimage)
        
        return qpixmap

    def remove_scraggly_bits(self, point_cloud, eps=0.003, min_points=10):
        labels = np.array(point_cloud.cluster_dbscan(eps=eps, min_points=min_points, print_progress=False))
        largest_cluster_label = np.bincount(labels[labels >= 0]).argmax()
        return point_cloud.select_by_index(np.where(labels == largest_cluster_label)[0])


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


    def generate_point_cloud(self):
        if not self.processed_images or not self.depth_images or not self.aruco_datas:
            print("No images or depth data available.")
            return

        # Initialize accumulated point cloud and reuse constants
        accumulated_point_cloud = o3d.geometry.PointCloud()
        camera_matrix_cached = camera_matrix()
        dist_coeffs_cached = dist_coeffs()
        
        for i, (rgb_image, depth_image, aruco_data) in enumerate(zip(self.processed_images, self.depth_images, self.aruco_datas)):
            point_cloud = self.depth_to_point_cloud(rgb_image, depth_image)
            point_cloud = self.remove_scraggly_bits(point_cloud, min_points=30)

            ids, corners = aruco_data[1], aruco_data[0]

            ## TODO FOR NOW IGNORING POINT CLOUDS WITH <2 ARCURO MARKERS BECAUSE THEY SUCK 
            if ids is None or len(ids) <= 1:
                continue

            objpoints, imgpoints = aruco_board().matchImagePoints(corners, ids)

            _, rvec, tvec = cv2.solvePnP(
                objectPoints=objpoints,
                imagePoints=imgpoints,
                cameraMatrix=camera_matrix_cached,
                distCoeffs=dist_coeffs_cached,
                flags=cv2.SOLVEPNP_ITERATIVE
            )
            
            # # Compute transformation matrix
            R, _ = cv2.Rodrigues(rvec)

            transformation_matrix = np.eye(4)
            transformation_matrix[:3, :3] = R
            transformation_matrix[:3, 3] = tvec.flatten()
            transformation_matrix_inverse = np.linalg.inv(transformation_matrix)

            print(tvec)

            # Apply transformation to all points at once
            points = np.asarray(point_cloud.points)  # Convert points to a NumPy array
            points_homogeneous = np.hstack((points, np.ones((points.shape[0], 1))))
            transformed_points = (transformation_matrix_inverse @ points_homogeneous.T).T[:, :3]

            point_cloud.points = o3d.utility.Vector3dVector(transformed_points)

            if i == 0:
                accumulated_point_cloud = point_cloud
            else:
                accumulated_point_cloud += point_cloud

        accumulated_point_cloud.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=0.10, max_nn=5))

        # Visualize the accumulated point cloud
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