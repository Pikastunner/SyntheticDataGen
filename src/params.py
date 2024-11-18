from cv2 import aruco
import numpy as np


### REPLACE WITH ARUCO BOARD YOU ARE USING (THE MORE POINTS THE BETTER)
def aruco_board():
    # Define the object points (3D coordinates) for each marker
    objectPoints = []  # List to hold the object points for each marker

    # Define positions for each marker in a list
    positions = [
        [[-0.01, 0.095, 0], [0.01, 0.095, 0], [0.01, 0.115, 0], [-0.01, 0.115, 0]],  # Marker 0
        [[0.095, 0.01, 0], [0.095, -0.01, 0], [0.115, -0.01, 0], [0.115, 0.01, 0]],  # Marker 1
        [[0.01, -0.095, 0], [-0.01, -0.095, 0], [-0.01, -0.115, 0], [0.01, -0.115, 0]],  # Marker 2
        [[-0.095, -0.01, 0], [-0.095, 0.01, 0], [-0.115, 0.01, 0], [-0.115, -0.01, 0]],  # Marker 3
        [[0.05813994098078374, 0.07579411100310487, 0], [0.0726474284010295, 0.06202701948922979, 0], [0.08641451991490459, 0.07653450690947555, 0], [0.07190703249465882, 0.09030159842335064, 0]],
        [[0.07579411100310487, -0.05813994098078374, 0], [0.06202701948922979, -0.07264742840102949, 0], [0.07653450690947554, -0.08641451991490458, 0], [0.09030159842335063, -0.07190703249465884, 0]],
        [[-0.05813994098078374, -0.07579411100310487, 0], [-0.0726474284010295, -0.06202701948922979, 0], [-0.08641451991490459, -0.07653450690947555, 0], [-0.07190703249465882, -0.09030159842335064, 0]],
        [[-0.07579411100310487, 0.05813994098078374, 0], [-0.06202701948922979, 0.07264742840102949, 0], [-0.07653450690947554, 0.08641451991490458, 0], [-0.09030159842335063, 0.07190703249465884, 0]],    # Calculate the object points based on their positions on the board
    ]
    for pos in positions:
        objectPoints.append(np.array(pos, dtype=np.float32))

    return aruco.Board(objPoints=np.array(objectPoints, dtype=np.float32), dictionary=aruco.getPredefinedDictionary(aruco.DICT_4X4_50), ids=np.array([0, 1, 2, 3, 4, 5, 6, 7]))


### REPLACE BOTH WITH YOUR OWN CALIBRATED INTRINSINCS 
def camera_matrix(): # depth camera intriniscs
    return np.array([[387.2332763671875, 0, 318.49847412109375],
                     [0, 387.2332763671875, 241.1577606201172],
                     [0, 0, 1]])


def camera_matrix_rgb():
    return np.array([[607.61181641, 0, 326.71447754],
                        [0, 607.38842773, 245.25457764],
                        [0, 0, 1]])


# def camera_matrix(): # depth camera intriniscs
#     return np.array([[385, 0, 320],
#                      [0, 385, 240],
#                      [0, 0, 1]], dtype=float)

# def camera_matrix_rgb():
#     return np.array([[605, 0, 325],
#                      [0, 605, 245],
#                      [0, 0, 1]], dtype=float)



def dist_coeffs():
    return np.zeros(5)
