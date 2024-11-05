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
#     return np.array([[607.61181641, 0, 326.71447754],
#         [  0,       607.38842773, 245.25457764],
#         [  0,          0,          1,       ]])

def camera_matrix():
    return np.array([[387.2332763671875, 0, 318.49847412109375],
                     [0, 387.2332763671875, 241.1577606201172],
                     [0, 0, 1]])



def dist_coeffs():
    return np.zeros(5)

# def dist_coeffs():
#     return np.array([ -0.3321, 0.1393, 0.0002, 0.0001, 0.0])