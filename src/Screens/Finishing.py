
import os
import sys
import cv2
from PyQt5.QtWidgets import (QWidget, QPushButton, QLabel, QVBoxLayout, QHBoxLayout)
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import Qt
import re

from Screens.Constants import OUTPUT_PATH
from pxr import Usd, UsdGeom, Vt, Gf
from pxr import Usd, UsdGeom, Gf, Vt, Sdf, UsdShade
import numpy as np
import json


from plyer.utils import platform
from plyer import notification

# Preprocessing Page
class FinishingScreen(QWidget):

    def update_triangle_mesh(self, triangle_mesh):
        self.mesh = triangle_mesh

    # This function is called when FinishingScreen made visible    
    def update_variables(self, num_images, output_path):
        self.num_images = num_images
        self.output_path = output_path

        FinishingScreen.convert_mesh_to_usd(self.mesh, usd_file_path=self.output_path+"/mesh_usd.usda")
        self.generate_images()

        # self.setup_gui()
        notification.notify(
            title='Rendering Finished',
            message='Open the application to view your images...',
            app_name='SyntheticDataGen',
        )

        # self.setup_gui()


    @staticmethod
    def convert_mesh_to_usd(open3d_mesh, usd_file_path="./_output/mesh_usd.usda"):
        # Ensure the directory exists
        output_directory = os.path.dirname(usd_file_path)
        if not os.path.exists(output_directory):
            os.makedirs(output_directory)

        # Create a new USD stage
        stage = Usd.Stage.CreateNew(usd_file_path)

        # Create a new mesh in the stage
        mesh_prim = UsdGeom.Mesh.Define(stage, '/GeneratedMesh')

        # Get vertices and triangles from the Open3D mesh
        vertices = np.asarray(open3d_mesh.vertices)
        triangles = np.asarray(open3d_mesh.triangles)

        # Set vertex points in the USD mesh
        mesh_prim.GetPointsAttr().Set(Vt.Vec3fArray(vertices.tolist()))

        # Flatten the triangle indices and set face vertex indices
        face_vertex_indices = triangles.flatten().tolist()
        mesh_prim.GetFaceVertexIndicesAttr().Set(Vt.IntArray(face_vertex_indices))

        # Set face counts assuming all faces are triangles
        face_counts = [3] * len(triangles)
        mesh_prim.GetFaceVertexCountsAttr().Set(Vt.IntArray(face_counts))

        # Check if the Open3D mesh has vertex colors
        if open3d_mesh.has_vertex_colors():
            vertex_colors = np.asarray(open3d_mesh.vertex_colors)
            # Convert vertex colors to a format compatible with USD (Gf.Vec3f)
            color_values = [Gf.Vec3f(*color) for color in vertex_colors]

            # Create the display color attribute with varying interpolation
            mesh_prim.CreateDisplayColorAttr().Set(Vt.Vec3fArray(color_values))
            mesh_prim.GetDisplayColorPrimvar().SetInterpolation(UsdGeom.Tokens.vertex)
            
        # Save the stage
        stage.GetRootLayer().Save()
        
    def generate_images(self, obj_usd_location=None):
        import subprocess  # Runs as separate process to avoid errors
        subprocess.run(['python', './src/Screens/Generator.py', self.output_path, str(self.num_images)], stdout=None, stderr=None, text=True)


    def __init__(self, parent):
        super().__init__()
        self.parent = parent

        self.output_path = str()
        self.image_index = 0
        self.image_info = QLabel()
        self.large_image_region = QLabel()
        self.processed_images = None
        self.small_image_one = QLabel()
        self.small_image_dim = None


    def setup_gui(self):
        # Set up the initial container
        text_layout = QVBoxLayout()
        text_area = QWidget(objectName="FinScreenTextArea")

        # Working within the initial container
        text_container_layout = QVBoxLayout()
        title = QLabel("Synthetic Data Generation Complete", objectName="FinScreenTitle")

        location_text = QLabel(f"View data in the following <a href='{self.output_path}'>directory</a>", objectName="FinScreenLocationText")
        location_text.setOpenExternalLinks(False)        
        location_text.linkActivated.connect(self.open_directory)

        data_caption = QLabel("Preview of generated data", objectName="FinScreenDataCaption")
        text_container_layout.addWidget(title)
        text_container_layout.addWidget(location_text)
        text_container_layout.addWidget(data_caption)
        text_area.setLayout(text_container_layout)

        # Update changes to initial container
        text_layout.addWidget(text_area)

        # Set up the initial container
        generated_data_layout = QHBoxLayout()
        generated_data_area = QWidget(objectName="FinScreenGenDataArea")

        # Working within the initial container
        total_layout = QHBoxLayout()
        total_layout.setSpacing(10)
        large_image_area = QWidget(objectName="FinScreenLargeImageArea")
        large_image_area.setStyleSheet("margin-left: 15px;")

        large_image_layout = QVBoxLayout()
        large_image_layout.setContentsMargins(0, 0, 0, 0)
        large_image_layout.setSpacing(0)

        self.large_image_region = QLabel()
        self.processed_images = self.load_output_images(output_path=self.output_path+"/coco_data/RenderProduct_Replicator/")
        print(self.output_path+"/coco_data/RenderProduct_Replicator/")
        self.image_index = 0

        available_width = int(self.large_image_region.width() * 0.80)
        available_height = int(self.large_image_region.height() * 0.85)
        self.large_image_dim = (available_width, available_height)

        self.apply_image(self.large_image_region, 0, self.large_image_dim)

        navigation_image_region = QWidget(objectName="FinScreenNavigationRegion")

        navigation_image_region_layout = QVBoxLayout()
        
        self.image_info = QLabel(objectName="FinImageInfo")
        self.update_image_info()

        next_button = QPushButton("Next")
        next_button.setFixedSize(80, 20)
        next_button.clicked.connect(self.next_image)

        navigation_image_region_layout.addWidget(self.image_info, alignment=Qt.AlignHCenter)
        navigation_image_region_layout.addWidget(next_button, alignment=Qt.AlignHCenter)

        large_image_layout.addWidget(self.large_image_region, 85)
        large_image_layout.addWidget(navigation_image_region, 15)

        navigation_image_region.setLayout(navigation_image_region_layout)

        large_image_area.setLayout(large_image_layout)

        small_image_area = QWidget(objectName="FinSmallImageArea")
        small_image_area.setStyleSheet("margin-right: 5px;")

        small_image_layout = QVBoxLayout()
        small_image_layout.setContentsMargins(0, 0, 0, 0)
        small_image_layout.setSpacing(0)
        
        self.small_image_region = QWidget(objectName="FinSmallImageRegion")
        small_image_region_layout = QVBoxLayout()
        small_image_region_layout.setContentsMargins(0, 0, 0, 0)
        small_image_region_layout.setSpacing(10)

        small_available_width = int(self.small_image_region.width() / 3 * 0.80)
        small_available_height = int(self.small_image_region.height() / 3 * 0.85)
        self.small_image_dim = (small_available_width, small_available_height)

        # self.small_image_one = QLabel()
        self.apply_image(self.small_image_one, 1, self.small_image_dim)
        self.small_image_two = QLabel()
        self.apply_image(self.small_image_two, 2, self.small_image_dim)
        self.small_image_three = QLabel()
        self.apply_image(self.small_image_three, 3, self.small_image_dim)
        
        small_image_region_layout.addWidget(self.small_image_one)
        small_image_region_layout.addWidget(self.small_image_two)
        small_image_region_layout.addWidget(self.small_image_three)

        self.small_image_region.setLayout(small_image_region_layout)

        small_image_layout.setAlignment(Qt.AlignTop)
        small_image_layout.addWidget(self.small_image_region)

        small_image_area.setLayout(small_image_layout)

        total_layout.addWidget(large_image_area, 75)
        total_layout.addWidget(small_image_area, 25)

        generated_data_area.setLayout(total_layout)
        generated_data_layout.addWidget(generated_data_area)

        # Set up the initial container
        bottom_area = QHBoxLayout()      
        bottom_widget = QWidget()
        bottom_widget.setObjectName("BottomWidget")
        bottom_layout = QHBoxLayout()
        bottom_widget.setLayout(bottom_layout)
        
        # Create next button with specified size and alignment
        finish_button = QPushButton("Finish")
        finish_button.setFixedSize(100, 30)
        finish_button.clicked.connect(self.exit_app)
        bottom_layout.addWidget(finish_button, 0, Qt.AlignRight | Qt.AlignBottom)

        bottom_area.addWidget(bottom_widget)

        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        main_layout.addLayout(text_layout, 25)
        main_layout.addLayout(generated_data_layout, 70)
        main_layout.addLayout(bottom_area, 5)

        self.setLayout(main_layout) 


    def exit_app(self):
        sys.exit()

    def open_directory(self):
        path = self.parent.widget(self.parent.currentIndex() - 1).directory_input.text()
        # Depending on the OS, open the file explorer to the specified directory
        if os.name == 'nt':  # Windows
            os.startfile(path)

    def load_output_images(self, output_path=None):
        folder_path = output_path if output_path else "./_output"
        rgb_image_files = self.get_files_starting_with(folder_path, 'rgb')
        if rgb_image_files:
            rgb_images = [cv2.cvtColor(cv2.imread(filename), cv2.COLOR_BGR2RGB) for filename in rgb_image_files]
            return rgb_images
    
    def get_files_starting_with(self, folder_path, prefix):
        files = []
        for file in os.listdir(folder_path):
            if re.search(rf"{prefix}", file):
                files.append(os.path.join(folder_path, file))
        return files

    def numpy_to_qimage(self, img):
        height, width, channel = img.shape
        bytes_per_line = channel * width
        qimage = QImage(img.data, width, height, bytes_per_line, QImage.Format_RGB888)
        return qimage

    def apply_image(self, image: QLabel, index: int, dimensions: tuple):
        if self.processed_images is None:
            image.setFixedSize(500, 500)
            image.setPixmap(QPixmap(500, 500))
            return

        qimage = self.numpy_to_qimage(self.processed_images[(self.image_index + index) % len(self.processed_images)])
        image.setPixmap(QPixmap.fromImage(qimage))
        # Calculate the maximum size for the image based on the available space
        available_width = dimensions[0]
        available_height = dimensions[1]
        image_width = qimage.width()
        image_height = qimage.height()
        aspect_ratio = image_width / image_height
        if aspect_ratio > available_width / available_height:
            scaled_width = available_width
            scaled_height = int(scaled_width / aspect_ratio)
        else:
            scaled_height = available_height
            scaled_width = int(scaled_height * aspect_ratio)

        # Set the size of the QLabel and the image
        image.setFixedSize(scaled_width, scaled_height)
        image.setPixmap(QPixmap.fromImage(qimage.scaled(scaled_width, scaled_height)))

    def next_image(self):
        self.image_index = (self.image_index + 1) % len(self.processed_images)
        self.apply_image(self.large_image_region, 0, self.large_image_dim)
        self.apply_image(self.small_image_one, 1, self.small_image_dim)
        self.apply_image(self.small_image_two, 2, self.small_image_dim)
        self.apply_image(self.small_image_three, 3, self.small_image_dim)
        self.update_image_info()  # Update the label text
        return
    
    def update_image_info(self):
        if self.processed_images is None:
            self.image_info.setText(f"Image #{self.image_index + 1} of 0")
            return 
        self.image_info.setText(f"Image #{self.image_index + 1} of {len(self.processed_images)}")

