import pyrealsense2 as rs
import numpy as np
import cv2
import os
from PyQt5.QtCore import QThread, pyqtSignal
from time import sleep

OUTPUT_PATH = "input_images"

RESOLUTION = (640, 480)

class CameraWorker(QThread):
    frameCaptured = pyqtSignal(np.ndarray, np.ndarray)  # Signal to emit captured frames
    photoSaved = pyqtSignal(str, str)       # Signal to emit photo file paths when saved

    def __init__(self, parent=None):
        super(CameraWorker, self).__init__(parent)
        self.pipeline = None
        self.running = False

    def run(self):
        """Run the camera capture in the thread"""
        try:
            self.pipeline = rs.pipeline()
            config = rs.config()
            config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
            config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
            self.pipeline.start(config)

            self.running = True
            while self.running:
                try:
                    frames = self.pipeline.wait_for_frames(10000)  # 10-second timeout
                    align_to = rs.stream.color  # Align depth to RGB
                    align = rs.align(align_to)

                    aligned_frames = align.process(frames)
                    color_frame = aligned_frames.get_color_frame()
                    depth_frame = aligned_frames.get_depth_frame()
                    if not color_frame or not depth_frame:
                        continue
                    
                    # Convert RealSense color frame to NumPy array
                    color_image = np.asanyarray(color_frame.get_data())
                    color_image = color_image[:, :, ::-1]

                    depth_frame = np.asanyarray(depth_frame.get_data())
                    
                    # Emit the frame to the UI thread
                    self.frameCaptured.emit(color_image, depth_frame)
                except RuntimeError as e:
                    print(f"Warning: {e}. Retrying...")
                    sleep(1)  # Wait 1 second before retrying
        finally:
            if self.pipeline:
                self.pipeline.stop()

    def take_photo(self):
        """Take a photo and save it to disk"""
        try:
            if not self.pipeline:
                print("Pipeline is not running. Cannot take photo.")
                return

            color_image, depth_image = self.capture_frames()
            if color_image is not None and depth_image is not None:
                rgb_filename, depth_filename, _ = save_images(color_image, depth_image)
                self.photoSaved.emit(rgb_filename, depth_filename)
            else:
                print("Failed to capture image.")
        except Exception as e:
            print(f"Error capturing photo: {e}")

    def stop(self):
        """Stop the camera thread"""
        self.running = False
        self.wait()  # Wait for the thread to exit

    def capture_frames(self):
        """Capture aligned RGB and depth frames from the RealSense camera"""
        frames = self.pipeline.wait_for_frames()
        align_to = rs.stream.color  # Align depth to RGB
        align = rs.align(align_to)

        aligned_frames = align.process(frames)
        aligned_depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()

        if not aligned_depth_frame or not color_frame:
            print("Error: Could not retrieve aligned frames.")
            return None, None

        color_image = np.asanyarray(color_frame.get_data())
        aligned_depth_image = np.asanyarray(aligned_depth_frame.get_data())

        return color_image, aligned_depth_image

def save_images(color_image, depth_image):
    """Save the RGB and depth images to disk"""
    if not os.path.exists(OUTPUT_PATH):
        os.makedirs(OUTPUT_PATH)

    # Generate filenames
    rgb_image_filename = os.path.join(OUTPUT_PATH, f'rgb_image_{create_image_id()}.png')
    depth_image_colormap_filename = os.path.join(OUTPUT_PATH, f'depth_image_colormap_{create_image_id()}.png')
    depth_image_filename = os.path.join(OUTPUT_PATH, f'depth_image_{create_image_id()}.png')

    # Apply colormap to depth image for visualization
    depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

    # Save the RGB and depth images
    cv2.imwrite(rgb_image_filename, color_image)
    cv2.imwrite(depth_image_colormap_filename, depth_colormap)
    cv2.imwrite(depth_image_filename, depth_image)


    return rgb_image_filename, depth_image_filename, depth_image_colormap_filename

def create_image_id():
    """Create a unique image ID based on the number of files in the output folder"""
    files = os.listdir(OUTPUT_PATH)
    return int(len(files) / 3)
    
# Function to check if camera is connected
def is_camera_connected():
   # return True
    try:
        # Create a context object to manage devices
        context = rs.context()

        # Get a list of connected devices
        devices = context.query_devices()

        # Check if any devices are connected
        if len(devices) > 0:
            print(f"Connected devices: {len(devices)}")
            return True
        else:
            print("No RealSense devices connected.")
            return False
    except Exception as e:
        print(f"Error while checking devices: {str(e)}")
        return False
