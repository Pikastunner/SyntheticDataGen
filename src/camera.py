'''
File to handle camera settings and photo taking
'''

import pyrealsense2 as rs
import numpy as np
import cv2
import os

OUTPUT_PATH = "input_images"

# Function to tell whether camera is connected
def is_camera_connected():
    context = rs.context()
    if len(context.devices) == 0:
        return False
    return True

# Function to return preview of camera
def preview_image():
    pipeline = initialize_camera()
    try:
        color_image, depth_image = capture_frames(pipeline)
        return color_image, depth_image
    finally:
        release_camera(pipeline)

# Function to take a photo and then save it to folder
def take_photo():
    output = capture_and_save_single_frame()
    if output:
        print(f"RGB image saved at: {output['rgb_image']}")
        print(f"Depth image saved at: {output['depth_image']}")
    else:
        print("Failed to capture images.")



# Helper function to create unique id
def create_image_id():
    files = os.listdir(OUTPUT_PATH)
    return int(len(files) / 2)

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

# Function to align depth to RGB
def align_frames(frames):
    align_to = rs.stream.color  # Align depth to RGB
    align = rs.align(align_to)

    # Perform the alignment
    aligned_frames = align.process(frames)
    
    # Get aligned frames
    aligned_depth_frame = aligned_frames.get_depth_frame()  # Aligned depth frame
    color_frame = aligned_frames.get_color_frame()          # Color frame remains the same
    
    if not aligned_depth_frame or not color_frame:
        print("Error: Could not retrieve aligned frames.")
        return None, None
    
    # Convert images to numpy arrays
    try:
        color_image = np.asanyarray(color_frame.get_data())
        aligned_depth_image = np.asanyarray(aligned_depth_frame.get_data())
    except ValueError as e:
        print(f"Error converting frames to numpy arrays: {e}")
        return None, None

    return color_image, aligned_depth_image

# Function to capture aligned RGB and depth frames
def capture_frames(pipeline):
    try:
        # Wait for frames (both depth and color)
        frames = pipeline.wait_for_frames()

        # Align depth to color image
        color_image, aligned_depth_image = align_frames(frames)
        
        if color_image is None or aligned_depth_image is None:
            print("Error: Could not align frames.")
            return None, None
        
        return color_image, aligned_depth_image
    except RuntimeError as e:
        print(f"Runtime error while waiting for frames: {e}")
        return None, None

# Function to save images to a folder
def save_images(color_image, depth_image, output_folder=OUTPUT_PATH):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Generate filenames
    rgb_image_filename = os.path.join(output_folder, 'rgb_image_' + str(create_image_id()) + '.png')
    depth_image_filename = os.path.join(output_folder, 'depth_image_colormap_' + str(create_image_id()) + '.png')
    
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
