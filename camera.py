import pyrealsense2 as rs
import numpy as np
import cv2

# Return boolean on if the camera is connected
def is_camera_connected() -> bool:
    # Create a context object, which manages the RealSense devices
    context = rs.context()

    # Get a list of all connected devices
    connected_devices = context.query_devices()

    if len(connected_devices) == 0:
        # No RealSense camera is connected
        return False
    else:
        # RealSense camera is connected
        return True

# Take a photo and place it in a folder
def take_photo(x_resolution = 640, y_resolution = 480) -> bool:
    # Initialize pipeline
    pipeline = rs.pipeline()

    # Create a config object and enable both color (RGB) and depth streams
    config = rs.config()
    config.enable_stream(rs.stream.color, x_resolution, y_resolution, rs.format.bgr8, 30)  # RGB stream
    config.enable_stream(rs.stream.depth, x_resolution, y_resolution, rs.format.z16, 30)   # Depth stream

    # Start the pipeline
    pipeline.start(config)

    # Wait for a coherent pair of frames: color
    frames = pipeline.wait_for_frames()
    color_frame = frames.get_color_frame()

    # Convert image to numpy array
    color_image = np.asanyarray(color_frame.get_data())

    # Save the image as a PNG file
    cv2.imwrite('realsense_image.png', color_image)

    # Stop the pipeline
    pipeline.stop()

    # Cleanup the window
    cv2.destroyAllWindows()
