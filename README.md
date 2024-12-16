# **SyntheticDataGen**

This project converts a series of images into objects and generates scenes from that object.

## **About**
SyntheticDataGen is a software tool designed for processing RGB and depth images to create 3D meshes and synthetic image datasets. It supports capturing images through an Intel RealSense camera or uploading existing photos. The generated 3D models can be used for scene generation in NVIDIA Omniverse, making it ideal for tasks in computer vision, robotics, and more.

## **Main Features**
- **Uploading/Capturing Images**:
  - Users can select between two options when inputting RGB and depth images: capturing the object using the in-app camera functionality (using the RealSense camera) and uploading photos saved elsewhere on the computer.
  - **Capture Option**: Users can use the RealSense camera to capture RGB and depth images. The feed can be switched between the RGB and depth image previews, and aruco markers are detected and displayed.
  - **Upload Option**: Users can browse and upload previously captured RGB and depth images from their files.

- **Review/Processing of Images**:
  - The RGB and depth images are converted into a 3D mesh.
  - **Background Removal**: Achieved using the `rembg[cuda]` library.
  - **Mesh Generation**: Utilizes `open3d`, `cv2`, `numpy`, `scikit-learn`, and other concurrency enabling libraries.
  - **Photo Review**: Users can review the images, delete unwanted ones, and view a loading animation with an estimated processing time.

- **Scene Generation**:
  - Users can upload the 3D model into NVIDIA Omniverse and perform scene generation using the IsaacSim engine. This allows for rotating the 3D mesh, taking snapshots, and forming a synthetic image dataset.
  - **Data Configuration**: Users can set the output directory and choose the number of synthetic images to generate. A size estimate of disk space used is displayed.
  - **Synthetic Image Navigation**: Users can navigate through generated images using the GUI and view data in the selected output directory where data is stored in COCO format.

## **Installation Guide**

### **Requirements**
Before installation, ensure that your system meets the following:
- **RAM**: 8GB or more.
- **GPU Memory**: 4GB or more.
- **Python**: The software was developed and tested on Python 3.10.11 on Windows. However, any version of Python 3.10 should work, as well as any Linux distributions.
- **Camera**: To make use of any camera features, users must have an Intel RealSense compatible camera.

### **Clone Repository**
1. Clone the Git project into your chosen directory:
   ```cmd
   git clone https://github.com/unsw-cse-comp99-3900/capstone-project-2024-t3-3900-W15A_CELERY.git
   cd capstone-project-2024-t3-3900-W15A_CELERY
   ```

### **Activate Virtual Environment (Optional)**
For a clean installation, it is recommended to use a Python virtual environment:
   ```cmd
   python -m venv .venv
   source .venv/bin/activate
   ```

### **Install Dependencies**
Install the required dependencies:
   ```cmd
   pip install -m requirements.txt
   ```

### **Running the Application**
To run the application, use the command:
   ```cmd
   python src/app.py
   ```

### **Website and Downloading Executable**
You can also download the executable version of SyntheticDataGen from the [website](#). This version is specifically useful for users who do not want to run the application from source. The executable will be updated periodically, so make sure to check for the latest version.
