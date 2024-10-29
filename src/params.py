from cv2 import aruco
import numpy as np

def aruco_board():
    # Define the object points (3D coordinates) for each marker
    objectPoints = []  # List to hold the object points for each marker

    # Define positions for each marker in a list
    positions = [
        [[-0.01, 0.095, 0], [0.01, 0.095, 0], [0.01, 0.115, 0], [-0.01, 0.115, 0]],  # Marker 0
        [[0.095, 0.01, 0], [0.095, -0.01, 0], [0.115, -0.01, 0], [0.115, 0.01, 0]],  # Marker 1
        [[0.01, -0.095, 0], [-0.01, -0.095, 0], [-0.01, -0.115, 0], [0.01, -0.115, 0]],  # Marker 2
        [[-0.095, -0.01, 0], [-0.095, 0.01, 0], [-0.115, 0.01, 0], [-0.115, -0.01, 0]]   # Marker 3
    ]

    # Calculate the object points based on their positions on the board
    for pos in positions:
        objectPoints.append(np.array(pos, dtype=np.float32))

    # Create the ArUco board
    return aruco.Board(objPoints=np.array(objectPoints, dtype=np.float32), dictionary=aruco.getPredefinedDictionary(aruco.DICT_4X4_50), ids=np.array([0, 1, 2, 3]))


# def camera_matrix():
#     return  np.array([[606.86, 0, 321.847],
#                                        [0, 606.86, 244.995],
#                                        [0, 0, 1]])


def camera_matrix():
    return np.array([[616.0, 0, 320.0],
                     [0, 616.0, 240.0],
                     [0, 0, 1]])


def dist_coeffs():
    return np.zeros(5)

# def dist_coeffs():
#     return np.array([-0.3950, 0.1572, 0.0, 0.0, 0.0])