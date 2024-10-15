import pyrealsense2 as rs
import numpy as np
import cv2
import os

OUTPUT_PATH = "output_images"

# Helper function to create unique id
def create_image_id():
    files = os.listdir(OUTPUT_PATH)
    return int(len(files) / 2)

# Function to check if a RealSense camera is connected
def is_camera_connected():
    context = rs.context()
    if len(context.devices) == 0:
        return False
    return True

# Function to initialize and configure the RealSense camera pipeline
def initialize_camera():
    pipeline = rs.pipeline()
    config = rs.config()
    
    # Enable RGB and depth streams
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    
    # Start streaming
    try:
        pipeline.start(config)
    except Exception as e:
        print(f"Error starting camera stream: {e}")
        return None
    
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
def save_images(color_image, depth_image, output_folder='output_images'):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Generate filenames for the RGB and depth images
    rgb_image_filename = os.path.join(output_folder, 'rgb_image_' + str(create_image_id()) + '.png')
    depth_image_filename = os.path.join(output_folder, 'depth_image_colormap_' + str(create_image_id()) + '.png')
    depth_filename = os.path.join(output_folder, 'depth_image_' + str(create_image_id()) + '.png')

    # Apply colormap to depth image for visualization
    depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
    
    # Save the RGB and depth images
    cv2.imwrite(rgb_image_filename, color_image)
    cv2.imwrite(depth_image_filename, depth_colormap)
    cv2.imwrite(depth_filename, depth_image)
    
    return rgb_image_filename, depth_image_filename

# Function to release the camera and close any windows
def release_camera(pipeline):
    if pipeline is not None:
        pipeline.stop()
    cv2.destroyAllWindows()

# Function to preview the camera feed and take a photo when a key is pressed
def preview_and_capture(output_folder='output_images'):
    if not is_camera_connected():
        print("Camera not connected")
        return

    pipeline = initialize_camera()
    if pipeline is None:
        print("Failed to initialize camera")
        return
    
    try:
        while True:
            color_image, aligned_depth_image = capture_frames(pipeline)
            
            # Ensure the frames are valid before proceeding
            if color_image is None or aligned_depth_image is None:
                continue
            
            try:
                # Apply colormap to depth image for better visualization
                depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(aligned_depth_image, alpha=0.03), cv2.COLORMAP_JET)
            
                # Ensure the shapes of the images match before stacking
                if color_image.shape != depth_colormap.shape:
                    print(f"Warning: RGB and depth colormap shapes do not match. RGB shape: {color_image.shape}, Depth shape: {depth_colormap.shape}")
                    continue
                
                # Display RGB and depth images side by side
                images_combined = np.hstack((color_image, depth_colormap))
                
                # Show the combined preview
                cv2.imshow('Preview - Press "c" to capture, "q" to quit', images_combined)
            except ValueError as e:
                print(f"Error combining images: {e}")
                continue
            
            # Wait for keypress: 'c' to capture, 'q' to quit
            key = cv2.waitKey(1) & 0xFF
            if key == ord('c'):
                # Save images when 'c' is pressed
                rgb_filename, depth_filename = save_images(color_image, aligned_depth_image, output_folder)
                print(f"Captured and saved RGB image at: {os.path.join(output_folder, rgb_filename)}")
                print(f"Captured and saved Depth image at: {os.path.join(output_folder, depth_filename)}")
            elif key == ord('q'):
                # Exit the loop on 'q'
                print("Exiting preview...")
                break

    finally:
        release_camera(pipeline)

# Example usage
if __name__ == "__main__":
    preview_and_capture()
