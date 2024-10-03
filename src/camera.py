'''
File to handle camera settings and photo taking
'''

import pyrealsense2 as rs
import numpy as np
import cv2
import os

OUTPUT_PATH = "input_images"

def create_photo_id():
    files = os.listdir(OUTPUT_PATH)
    return int(len(files) / 2)

# Function to tell whether camera is connected
def is_camera_connected():
    context = rs.context()
    if len(context.devices) == 0:
        return False
    return True

# Function to take a photo and then save it to folder
def take_photo():
    output = capture_and_save_single_frame()
    if output:
        print(f"RGB image saved at: {output['rgb_image']}")
        print(f"Depth image saved at: {output['depth_image']}")
    else:
        print("Failed to capture images.")

# Function to initialize and configure the RealSense camera pipeline
def initialize_camera():
    pipeline = rs.pipeline()
    config = rs.config()
    
    # Enable RGB and depth streams
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    
    # Start streaming
    pipeline.start(config)
    
    return pipeline

# Function to capture RGB and depth frames
def capture_frames(pipeline):
    try:
        # Wait for a coherent pair of frames: depth and color
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()
        
        if not depth_frame or not color_frame:
            return None, None
        
        # Convert frames to numpy arrays
        color_image = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(depth_frame.get_data())
        
        return color_image, depth_image
    except RuntimeError as e:
        print(f"Runtime error while waiting for frames: {e}")
        return None, None

# Function to save images to a folder
def save_images(color_image, depth_image, output_folder=OUTPUT_PATH):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Generate filenames
    rgb_image_filename = os.path.join(output_folder, 'rgb_image_' + str(create_photo_id()) + '.png')
    depth_image_filename = os.path.join(output_folder, 'depth_image_colormap_' + str(create_photo_id()) + '.png')
    
    # Apply colormap to depth image for visualization
    depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
    
    # Save the RGB and depth images
    cv2.imwrite(rgb_image_filename, color_image)
    cv2.imwrite(depth_image_filename, depth_colormap)
    
    return rgb_image_filename, depth_image_filename

# Function to release the camera and close any windows
def release_camera(pipeline):
    pipeline.stop()
    cv2.destroyAllWindows()

# Example function to capture and save a single frame (to be used in a frontend or other scripts)
def capture_and_save_single_frame(output_folder=OUTPUT_PATH):
    pipeline = initialize_camera()
    
    try:
        color_image, depth_image = capture_frames(pipeline)
        if color_image is not None and depth_image is not None:
            rgb_filename, depth_filename = save_images(color_image, depth_image, output_folder)
            return {'rgb_image': rgb_filename, 'depth_image': depth_filename}
        else:
            return None
    finally:
        release_camera(pipeline)

'''
# Example usage
if __name__ == "__main__":
    take_photo()
'''