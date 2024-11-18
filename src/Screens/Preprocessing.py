import os
import cv2
import numpy as np
from PyQt5.QtWidgets import (QWidget, QLineEdit, QPushButton, QLabel, QVBoxLayout, QHBoxLayout, 
                             QFileDialog, QMessageBox, QSizePolicy, QMainWindow, QCheckBox, QComboBox, QSpacerItem, QSpinBox, QDoubleSpinBox)
from PyQt5.QtGui import QImage, QPixmap,QIcon
from PyQt5.QtCore import Qt, QThread, QSize, QPoint, QRect
import cv2.aruco as aruco
from rembg import remove
from PIL import Image

from params import *

import concurrent.futures

import open3d as o3d

import concurrent.futures
import time
from scipy.spatial import Delaunay

from sklearn.neighbors import NearestNeighbors
from joblib import Parallel, delayed

# Preprocessing Page
class PreprocessingScreen(QWidget):

    ############################################################
            # CALLED WHEN WE FIRST SWITCH TO SCREEN 
            # (PUT STUFF HERE THAT DOESN'T NEED TO BE 
            # CALLED ON IMMEDIATE STARTUP)
    ############################################################
    def update_variables(self, rgb_filenames, depth_filenames):
        self.processed_images, self.depth_images, self.aruco_datas = self.process_images(rgb_filenames, depth_filenames)
        self.image_index = 0

        ## ANNOTATE THE IMAGES WITH THE BOARD CENTRE
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
        with concurrent.futures.ThreadPoolExecutor() as executor:
            tasks = [
                (i, self.processed_images[i], self.aruco_datas[i][0], self.aruco_datas[i][1])
                for i in range(len(self.processed_images))
            ]
            
            # Execute the tasks concurrently and collect the results
            results = executor.map(lambda args: PreprocessingScreen.annotate_image(*args), tasks)
            # Append valid results to annotated_images (excluding None)
            self.annotated_images.extend([result for result in results if result is not None])
            
        ## DISPLAY THE FIRST IMAGE IN THE GUI
        qimage = self.numpy_to_qimage(self.annotated_images[self.image_index])
        self.background_image.setPixmap(QPixmap.fromImage(qimage))
        self.background_image.setScaledContents(True)  # Allow the pixmap to scale with the label
        self.background_image.setGeometry(self.rect())  # Make QLabel cover the whole widget
        self.background_image.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)  # Set size policy
        self.background_image_info.setText(f"Image #1 of {len(self.processed_images)}")

        ## GET POINT CLOUD
        self.accumulated_point_cloud = self.generate_point_cloud()

        print("Initial point cloud has been generated...")

        ## THIS FILE CONTAINS ALL THE OUTPUT AND CLEARS EACH TIME WE RUN THE PROGRAM
        import shutil
        if os.path.exists("./_output"):
            # Remove the directory and all of its contents
            shutil.rmtree("./_output")
            print(f"Directory /_output/ and all its contents have been removed.")
        # Recreate the empty directory
        os.makedirs("./_output")

        o3d.io.write_point_cloud("./_output/pcl.pcd",  self.accumulated_point_cloud)     


        ## GENERATE MESH HANDLES THINGS FROM HERE ON
        self.generate_mesh()

    ############################################################
            # GUI BEHAVIOUR/DISPLAY AND CLASS VARS
    ############################################################
    def __init__(self, parent):
        super().__init__()

        self.parent = parent

        self.accumulated_point_cloud = o3d.geometry.PointCloud()
        self.triangle_mesh = o3d.geometry.TriangleMesh()

        self.o3d_visualizer = None

        self.settings_window = QMainWindow()

        # Title Section
        title_area = QWidget(objectName="PreprocessingTitleArea")
        title_layout = QVBoxLayout(title_area)
        title_layout.addWidget(QLabel("Preprocessing", objectName="PreprocessingLabel"))
        title_layout.addWidget(QLabel("Background Removal & 3D Interface Generation", objectName="Label2"))

        # Preprocessing Results Section
        preprocessing_results_area = QWidget(objectName="PreprocessingResultsArea")
        preprocessing_area_layout = QHBoxLayout(preprocessing_results_area)
        preprocessing_area_layout.setContentsMargins(0, 0, 0, 0)
        preprocessing_area_layout.setSpacing(0)

        # Background Section
        background_section = QWidget(objectName="PreprocessingBackgroundSection")
        background_section.setStyleSheet("margin-top: 15px; margin-left: 15px; margin-right: 7.5px;")
        background_layout = QVBoxLayout(background_section)
        background_layout.addWidget(QLabel("Review removed background", objectName="PreprocessingTitle"), 10)

        self.background_image = QLabel()
        self.processed_images = []
        self.annotated_images = []
        self.background_image_info = QLabel(f"Image #1 of {len(self.processed_images)}", objectName="PreprocessingBackgroundImageInfo")
        self.background_image_info.setAlignment(Qt.AlignCenter)

        next_button = QPushButton("Next")
        next_button.setFixedSize(110, 45)
        next_button.clicked.connect(self.move_to_next)

        center_widget = QWidget()
        center_layout = QVBoxLayout(center_widget)
        center_layout.addWidget(self.background_image_info, alignment=Qt.AlignHCenter)
        center_layout.addWidget(next_button, alignment=Qt.AlignHCenter)

        background_layout.addWidget(self.background_image, 90)
        background_layout.addWidget(center_widget)

        # Graphical Interface Section
        graphical_interface_section = QWidget(objectName="PreprocessingGraphInterfaceSection")
        graphical_interface_section.setStyleSheet("margin-top: 15px; margin-left: 7.5px; margin-right: 15px;")
        graphical_interface_layout = QVBoxLayout(graphical_interface_section)

        graphical_interface_layout.addWidget(QLabel("Graphical 3D interface of input image", objectName="PreprocessingGraphicalInterface"), 10)

        # Graphical image label
        self.graphical_interface_image = QLabel(objectName="PreprocessingGraphicalInterfaceImage")
        self.graphical_interface_image.setScaledContents(True)  # Scale to fit the label
        self.graphical_interface_image.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)  # Set size policy

        # Create the overlay button with icon
        icon_button = QPushButton(objectName = "Slider")
        icon_button.setIcon(QIcon("./src/Icons/slider_dark.svg"))  # Set the SVG icon
        icon_button.setIconSize(QSize(15, 15))  # Set size of the icon (optional)
        icon_button.clicked.connect(self.view_settings_window)  # Connect to the new window function
        icon_button.setFixedSize(55,45)

        # Fullscreen button
        fs_button = QPushButton("Fullscreen")
        fs_button.setFixedSize(140, 45)
        fs_button.clicked.connect(self.view_3d_interface)

        # Create a horizontal layout to place buttons on the same row
        button_layout = QHBoxLayout()
        button_layout.addWidget(icon_button)
        button_layout.addWidget(fs_button)

        # Create a widget to hold the buttons and image
        button_widget = QWidget()
        button_widget.setLayout(button_layout)

        center_widget_preview = QWidget()
        center_layout_preview = QVBoxLayout(center_widget_preview)
        center_layout_preview.addWidget(QLabel("\n"), alignment=Qt.AlignHCenter)
        center_layout_preview.addWidget(button_widget, alignment=Qt.AlignHCenter)

        graphical_interface_layout.addWidget(self.graphical_interface_image, 86)
        graphical_interface_layout.addWidget(center_widget_preview)
        
        preprocessing_area_layout.addWidget(background_section, 45)
        preprocessing_area_layout.addWidget(graphical_interface_section, 45)

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
        main_layout.addWidget(preprocessing_results_area, 80)
        main_layout.addWidget(navigation_area, 10)


        ### NOTE: THESE ARE SETTINGS THAT ARE BE USED IN SETTINGS PANEL
        self.enable_smoothing = False
        self.reconstruction_methods = ["Alpha Shapes", "Poisson Reconstruction"]
        self.reconstruction_method_default = "Alpha Shapes"

        # These settings should only appear when the method is poisson
        self.normal_estimation_neighbours = 30
        self.normal_estimation_radius = 0.05
        # This option should appear when the above is enabled
        self.poisson_density_filter = 9
        self.poisson_depth = 11

        # Options for alpha reconstruction
        self.alpha_detail = 0.1

        
    def view_3d_interface(self):
        coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
        o3d.visualization.draw_geometries([self.triangle_mesh, coordinate_frame])
        
    # def select_directory(self):
    #     directory = QFileDialog.getExistingDirectory(self, "Select Directory")
    #     if directory:
    #         self.directory_input.setText(directory)

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
            # STEP 1: LOAD/PROCESS DEPTH/RGB/ARUOCO IMAGES
    ############################################################

    @staticmethod
    def annotate_image(i, img, corners, ids):
        if len(ids) < 2:
            return None
        objpoints, imgpoints = aruco_board().matchImagePoints(corners, ids)

        if objpoints is None or imgpoints is None or not objpoints.any() or not imgpoints.any():
            return None
        if len(objpoints) < 4 or len(imgpoints) < 4:
            return None

        _, rvec, tvec = cv2.solvePnP(objectPoints=objpoints, imagePoints=imgpoints, cameraMatrix=camera_matrix(), distCoeffs=dist_coeffs(), flags=cv2.SOLVEPNP_ITERATIVE)
        return cv2.drawFrameAxes(img.copy(), cameraMatrix=camera_matrix(), distCoeffs=dist_coeffs(), rvec=rvec, tvec=tvec, thickness=3, length=0.02)


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
        parameters.cornerRefinementMethod = aruco.CORNER_REFINE_SUBPIX
        parameters.cornerRefinementMaxIterations = 40

        # Detect ArUco markers before background removal
        detector = aruco.ArucoDetector(dictionary, parameters)
        corners, ids, rejected = detector.detectMarkers(gray_image)

        if ids is not None:
            corners, ids, _, recovered = detector.refineDetectedMarkers(gray_image, board=aruco_board(), detectedCorners=corners, detectedIds=ids, rejectedCorners=rejected,cameraMatrix=camera_matrix(),distCoeffs=dist_coeffs())
            print(f"{len(ids)} aruco markers found")
            # print(ids)
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
        
        print("Processing images...")
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = {executor.submit(self.process_image, rgb_images[i], depth_images[i]): i 
                       for i in range(min(len(rgb_images), len(depth_images)))}
            for i, future in enumerate(concurrent.futures.as_completed(futures)):
                result = future.result()
                if result[0] is not None:
                    processed_images.append(result[0])
                    depth_returned.append(result[1])
                    aruco_returned.append(result[2])
        print ("Finished processing images...")

        return processed_images, depth_returned, aruco_returned

    
    def numpy_to_qimage(self, img):
        height, width, channel = img.shape
        bytes_per_line = channel * width
        qimage = QImage(img.data, width, height, bytes_per_line, QImage.Format_RGB888)
        return qimage

    ############################################################
            # STEP 2: CREATE A POINT CLOUD FROM IMAGES
    ############################################################

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
    

    @staticmethod
    def remove_scraggly_bits(point_cloud, eps=0.003, min_points=100):
        labels = np.array(point_cloud.cluster_dbscan(eps=eps, min_points=min_points, print_progress=True))
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
    
    ############################################################
            # STEP 3: CREATE A MESH FROM POINT CLOUD
    ############################################################



    def view_settings_window(self):
        # Create the settings window as a new QMainWindow
        self.settings_window = QMainWindow(self, objectName="SettingsWindow")
        self.settings_window.setWindowTitle("Mesh Generation Settings")
        self.settings_window.setWindowIcon(QIcon("./src/Icons/slider_dark.svg"))

        # Create a central widget for the settings window
        central_widget = QWidget()
        self.settings_window.setCentralWidget(central_widget)

        # Main layout
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(10, 10, 10, 10)  # Add some padding

        # Row for enabling/disabling smoothing
        self.add_checkbox_row(main_layout, "MLS Smoothing <sup>ⓘ</sup>", self.enable_smoothing, self.toggle_smoothing, tip="This applies Moving Least Squares smoothing through a PCL wrapper. <b>Note:</b> If you are using Microsoft Windows, WSL is required.")
        self.add_combo_box_row(main_layout, "Reconstruction Method", self.reconstruction_methods, self.reconstruction_method_default, self.update_reconstruction_method)
        self.add_spin_box_row(main_layout, "Normal Estimation Neighbors", self.normal_estimation_neighbours, 1, 1000, self.update_normal_estimation_neighbors)
        self.add_spin_box_row(main_layout, "Normal Estimation Radius", self.normal_estimation_radius, 0, 1, self.update_normal_estimation_radius, step=0.001)
        self.add_spin_box_row(main_layout, "Poisson Depth", self.poisson_depth, 1, 100, self.update_poisson_depth, step=1)
        self.add_spin_box_row(main_layout, "Poisson Density Filter", self.poisson_density_filter, 1, 100, self.update_poisson_density_filter, step=0.5)
        self.add_spin_box_row(main_layout, "Alpha Detail", self.alpha_detail, 0.01, 5.0, self.update_alpha_detail, step=0.01)

        # Regenerate button aligned to the bottom
        main_layout.addSpacerItem(QSpacerItem(20, 20, QSizePolicy.Minimum, QSizePolicy.Expanding))
        button = QPushButton("Regenerate")
        button.clicked.connect(self.generate_mesh)
        main_layout.addWidget(button, alignment=Qt.AlignBottom)

        # Set the main layout on the central widget
        central_widget.setLayout(main_layout)

        self.settings_window.show()

    def add_checkbox_row(self, layout, label, checked, on_change, tip=None):
        row_layout = QHBoxLayout()
        row_label = QLabel(label)
        row_label.setToolTip(tip)
        row_label.setFixedWidth(150)  # Fixed width for all labels
        row_checkbox = QCheckBox()
        row_checkbox.setChecked(checked)
        row_checkbox.stateChanged.connect(on_change)

        row_layout.addWidget(row_label)
        row_layout.addWidget(row_checkbox)
        row_layout.addStretch()
        row_layout.setAlignment(row_label, Qt.AlignVCenter)  # Vertical alignment
        row_layout.setAlignment(row_checkbox, Qt.AlignVCenter)  # Vertical alignment
        layout.addLayout(row_layout)

    def add_combo_box_row(self, layout, label, items, default_item, on_change):
        row_layout = QHBoxLayout()
        row_label = QLabel(label)
        row_label.setFixedWidth(150)  # Fixed width for all labels
        row_combo = QComboBox()
        row_combo.addItems(items)
        row_combo.setCurrentText(default_item)
        row_combo.currentTextChanged.connect(on_change)

        row_layout.addWidget(row_label)
        row_layout.addWidget(row_combo)
        row_layout.addStretch()
        row_layout.setAlignment(row_label, Qt.AlignVCenter)  # Vertical alignment
        row_layout.setAlignment(row_combo, Qt.AlignVCenter)  # Vertical alignment
        layout.addLayout(row_layout)

    def add_spin_box_row(self, layout, label, value, min_val, max_val, on_change, step=1, enabled=True):
        row_layout = QHBoxLayout()
        row_label = QLabel(label)
        row_label.setFixedWidth(150)  # Fixed width for all labels
        row_spinbox = QSpinBox() if step == 1 else QDoubleSpinBox()
        row_spinbox.setValue(value)
        row_spinbox.setRange(min_val, max_val)
        row_spinbox.setSingleStep(step)
        row_spinbox.setEnabled(enabled)
        row_spinbox.setFixedWidth(80)  # Set a fixed width for consistency
        row_spinbox.valueChanged.connect(on_change)

        row_layout.addWidget(row_label)
        row_layout.addWidget(row_spinbox)
        row_layout.addStretch()
        row_layout.setAlignment(row_label, Qt.AlignVCenter)  # Vertical alignment
        row_layout.setAlignment(row_spinbox, Qt.AlignVCenter)  # Vertical alignment
        layout.addLayout(row_layout)

    # Helper methods to update settings based on controls
    def toggle_smoothing(self, state):
        self.enable_smoothing = bool(state)
    def update_reconstruction_method(self, method):
        self.reconstruction_method_default = method            
    def update_normal_estimation_neighbors(self, value):
        self.normal_estimation_neighbours = value
    def update_poisson_depth(self, value):
        self.poisson_depth = value
    def update_poisson_density_filter(self, value):
        self.poisson_density_filter = value
    def update_alpha_detail(self, value):
        self.alpha_detail = value
    def update_normal_estimation_radius(self, value):
        self.normal_estimation_radius = value

    
    def generate_mesh(self):
        print("Generating new triangle mesh...")
        smoothing = self.enable_smoothing
        reconstruction_model = self.reconstruction_method_default
        point_cloud = self.accumulated_point_cloud

        ## IF SMOOTHING ENABLED CALL CPP UTILITY IN LINUX ENVIRONMENT
        if (smoothing):
            print("Smoothing has been enabled. Calling smoothing utility...")
            unrefined_cloud = self.accumulated_point_cloud
            o3d.io.write_point_cloud("./_output/pcl.pcd", unrefined_cloud)
            from Utilities.caller import run_command
            run_command("./src/Utilities/mesh_smooth_utility ./_output/pcl.pcd ./_output/pcl.pcd")
            point_cloud = o3d.io.read_point_cloud("./_output/pcl.pcd")
            point_cloud.colors = unrefined_cloud.colors
        
        ## RECONSTRUCT MESH BASED ON WHICHEVER METHOD YOU HAVE SPECIFIED
        if (reconstruction_model == "Alpha Shapes"):
            self.triangle_mesh = self.generate_mesh_with_alpha_shapes(point_cloud)
        if (reconstruction_model == "Poisson Reconstruction"):
            self.triangle_mesh = self.generate_mesh_with_poisson(point_cloud)

        print("Done generating mesh...")

        ## VISUALIZE NEW MESH AND ENSURE CORRECT GUI SCALING
        self.o3d_visualizer = o3DVisualizer(self.triangle_mesh)
        self.graphical_interface_image.setPixmap(self.o3d_visualizer.capturePreview())
        self.graphical_interface_image.setFixedHeight(self.background_image.height()) # Match background image height


    def estimate_normals_neighborhood_reconstruction(self, pcd, radius=0.5):
        kdtree = o3d.geometry.KDTreeFlann(pcd)

        # Placeholder for normals
        normals = []

        # Iterate through each point in the point cloud
        for i in range(len(pcd.points)):
            # Find neighbors within the specified radius
            [_, idx, _] = kdtree.search_radius_vector_3d(pcd.points[i], radius)
            if len(idx) < 3:
                normals.append([0, 0, 1])  # Default normal if less than 3 neighbors
                continue

            # Extract the neighbor points and get eigenvalues/vectors
            neighbor_points = np.asarray(pcd.points)[idx, :]
            covariance_matrix = np.cov(neighbor_points.T)
            eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)

            # Normal is the eigenvector corresponding to the smallest eigenvalue
            normal = eigenvectors[:, np.argmin(eigenvalues)]

            # Ensure normal direction consistency
            if np.dot(normal, pcd.points[i]) < 0:
                normal = -normal

            normals.append(normal)
        normals = np.array(normals)
        pcd.normals = o3d.utility.Vector3dVector(normals)

    @staticmethod
    def mls_smooth_point(i, points, nbrs, n_neighbors):
        # print(i)
        point = points[i]
        _, indices = nbrs.radius_neighbors([point])
        neighbors = points[indices[0]]
        
        # Compute the centroid of the neighbors
        centroid = np.mean(neighbors, axis=0)
        
        # Compute the covariance matrix of the neighbors
        cov_matrix = np.cov(neighbors - centroid, rowvar=False)
        
        # Compute the eigenvectors and eigenvalues of the covariance matrix
        _, eigenvectors = np.linalg.eigh(cov_matrix)
        
        # The normal vector is the eigenvector corresponding to the smallest eigenvalue
        normal = eigenvectors[:, 0]
        
        # Project the point onto the plane defined by the neighbors
        point_on_plane = point - np.dot(point - centroid, normal) * normal
        
        return point_on_plane

    @staticmethod
    def mls_smooth_parallel(pcd, radius=0.01, n_neighbors=30, n_jobs=-1):
        # Convert point cloud to numpy array
        points = np.asarray(pcd.points)
        
        # Create a NearestNeighbors object to find neighbors
        nbrs = NearestNeighbors(n_neighbors=n_neighbors, radius=radius)
        nbrs.fit(points)
        
        # Parallelize the MLS smoothing using joblib
        smoothed_points = Parallel(n_jobs=n_jobs)(
            delayed(PreprocessingScreen.mls_smooth_point)(i, points, nbrs, n_neighbors) for i in range(len(points))
        )
        
        # Convert the smoothed points back into a point cloud
        smoothed_pcd = o3d.geometry.PointCloud()
        smoothed_pcd.points = o3d.utility.Vector3dVector(np.array(smoothed_points))
        
        return smoothed_pcd

    def generate_mesh_with_poisson(self, pcd):
        voxel_size = 0.001  # Adjust voxel size for desired level of reduction
        downsampled_pcd = pcd.voxel_down_sample(voxel_size)

        # Convert the point clouds to numpy arrays
        original_points = np.asarray(pcd.points)
        reduced_points = np.asarray(downsampled_pcd.points)

        # Get the colors from the original point cloud
        original_colors = np.asarray(pcd.colors)

        # Use NearestNeighbors to find the closest points between the original and reduced point clouds and assign colors
        nbrs = NearestNeighbors(n_neighbors=1, algorithm='auto').fit(original_points)
        _, indices = nbrs.kneighbors(reduced_points)
        reduced_colors = original_colors[indices.flatten()]
        downsampled_pcd.colors = o3d.utility.Vector3dVector(reduced_colors)

        # Perform MLS smoothing with parallelization and set the colors again
        smoothed_pcd = PreprocessingScreen.mls_smooth_parallel(downsampled_pcd, radius=0.017, n_neighbors=120, n_jobs=-1)
        smoothed_pcd.colors = downsampled_pcd.colors

        self.estimate_normals_neighborhood_reconstruction(smoothed_pcd, radius=self.normal_estimation_radius)

        # Create the Poisson surface reconstruction and filter out bottom 5% density areas
        poisson_mesh, poisson_densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(smoothed_pcd, depth=self.poisson_depth)
        densities = np.asarray(poisson_densities)
        threshold = np.percentile(densities, self.poisson_density_filter)
        triangles_to_keep = densities > threshold
        filtered_mesh = poisson_mesh.select_by_index(np.where(triangles_to_keep)[0])

        return filtered_mesh


    def generate_mesh_with_alpha_shapes(self, pcl):
        pcl_np = np.asarray(pcl.points)
        colors_np = np.asarray(pcl.colors)  # Get colors as well

        # Find the minimum z-coordinate value in the point cloud
        min_z = np.min(pcl_np[:, 2])

        # Shift all vertices so that the bottommost vertex is at z=0
        pcl_np[:, 2] -= min_z

        # Now proceed with the mesh generation as before
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
        mask = circumradius < (1 / self.alpha_detail)
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
        mesh.compute_vertex_normals()

        return mesh  # Temporary


    ############################################################
            # PAGE CONTROLS
    ############################################################
        
    def go_to_back_page(self):
        # par = self.parent.stacked_widget
        current_index = self.parent.currentIndex()
        if current_index > 0:
            self.parent.setCurrentIndex(current_index - 4)
        else:
            print("Already on the first page")

    def go_to_next_page(self):
        current_index = self.parent.currentIndex()
        if current_index < self.parent.count() - 1:
            self.parent.setCurrentIndex(current_index + 1)
            two_screens_ahead = self.parent.widget(self.parent.currentIndex() + 1)
            two_screens_ahead.update_triangle_mesh(self.triangle_mesh)
        else:
            print("Already on the last page")


########################################################################################################################
        ## THIS IS A HELPER CLASS; CALLING O3D VISUALIZE BLOCKS MAIN THREAD AND THIS PUTS ON SEPERATE THREAD
########################################################################################################################
class o3DVisualizer(QThread):
    def __init__(self, triangle_mesh):
        super().__init__()
        self.triangle_mesh = triangle_mesh

    def fullscreen(self):
        # Create the coordinate frame and display the 3D mesh in a non-blocking way
        coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
        o3d.visualization.draw_geometries([self.triangle_mesh, coordinate_frame])

    def capturePreview(self) -> QPixmap:
        # Initialize Open3D visualizer
        vis = o3d.visualization.Visualizer()
        vis.create_window(visible=False)  # Create an invisible window
        vis.add_geometry(self.triangle_mesh)

        # Set background color to black
        render_option = vis.get_render_option()
        render_option.background_color = np.asarray([0.0, 0.0, 0.0])
        
        # Capture the image
        vis.poll_events()
        vis.update_renderer()
        image = vis.capture_screen_float_buffer(True)
        
        # Convert to numpy array
        image = np.asarray(image)
        
        # Convert the float buffer (RGB) into an 8-bit format
        image = (image * 255).astype(np.uint8)
        
        # Convert to QPixmap
        height, width, _ = image.shape
        bytes_per_line = 3 * width
        qimage = QImage(image.data, width, height, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimage)
        
        vis.destroy_window()
        
        return pixmap


from PyQt5.QtWidgets import (QApplication)
import sys
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
    image_processing_widget = PreprocessingScreen()

    # Set the layout for the main window
    main_layout = QVBoxLayout(main_window)
    main_layout.addWidget(image_processing_widget)

    # Set layout
    main_window.setLayout(main_layout)
    main_window.show()

    sys.exit(app.exec_())
