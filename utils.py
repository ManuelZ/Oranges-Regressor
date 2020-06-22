import cv2
import numpy as np

SQUARES_LENGTH = 0.01 # meters
LENGTH, WIDTH = 0.24, 0.15 # meters
OUT_IMG_LENGTH, OUT_IMG_SIZE_WIDTH = int((LENGTH/WIDTH) * 1000), 1000 # px

def create_board(aruco_dict):
    """
    Create a board with one Aruco marker in each corner.
    """
    # Top left coordinates of the markers
    corners_x = np.array([0.0, WIDTH, WIDTH, 0.0])
    corners_y = np.array([0.0, 0.0, LENGTH, LENGTH])
    
    marker_corner_vec = []
    ids_vec = []
	
    for i,_ in enumerate(corners_x):
        marker_corners = np.zeros((4, 3), dtype=np.float32) # The third coordinate isn't used

        # Corners in CCW order: bottom left, bottom right, top right, top left
        marker_corners[0, 0] = corners_x[i]
        marker_corners[0, 1] = corners_y[i] + SQUARES_LENGTH
        
        marker_corners[1, 0] = corners_x[i] + SQUARES_LENGTH
        marker_corners[1, 1] = corners_y[i] + SQUARES_LENGTH
        
        marker_corners[2, 0] = corners_x[i] + SQUARES_LENGTH
        marker_corners[2, 1] = corners_y[i] 
        
        marker_corners[3, 0] = corners_x[i] 
        marker_corners[3, 1] = corners_y[i]
        
        marker_corner_vec.append(marker_corners)
        ids_vec.append(i+1)

    ids_vec = np.array(ids_vec, dtype=np.uint8)
    board = cv2.aruco.Board_create(marker_corner_vec, aruco_dict, ids_vec)
        
    im = cv2.aruco.drawPlanarBoard(board, (OUT_IMG_SIZE_WIDTH, OUT_IMG_LENGTH), marginSize=50, borderBits=1)
    cv2.imshow("", im)
    cv2.imwrite("board.png", im)
    cv2.waitKey(0)

aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
create_board(aruco_dict)