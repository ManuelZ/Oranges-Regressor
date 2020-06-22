import cv2
import numpy as np
from numpy import loadtxt

SQUARES_LENGTH = 0.03 # meters
LENGTH, WIDTH = 0.24, 0.15 # meters
OUT_IMG_LENGTH, OUT_IMG_SIZE_WIDTH = int((LENGTH/WIDTH) * 3000), 3000 # px
CAMERA_MATRIX = 'camera_matrix_infinity.csv'
DIST_COEFFS = 'dist_coeffs_infinity.csv'

# TODO: generate a custom dictionary
# cv::Ptr<cv::aruco::Dictionary> dictionary = cv::aruco::generateCustomDictionary(36, 5);

# TODO: take two images and try to do 3D vision

def create_board(aruco_dict, save=True):
    """
    Create a board with one Aruco marker in each corner.
    """
    # Top left coordinates of the markers in a common XY system of coordinates
    # (origin is in the bottom left)
    corners_x = np.array([WIDTH, WIDTH, 0.0, 0])
    corners_y = np.array([0.0, LENGTH, LENGTH, 0])
    
    marker_corner_vec = []
    ids_vec = []
	
    for i,_ in enumerate(corners_x):
        marker_corners = np.zeros((4, 3), dtype=np.float32) # The third coordinate isn't used

        # Corners of each marker need to be defined in CCW order: 
        # bottom left, bottom right, top right, top left
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
    if save:
        cv2.imshow("", im)
        cv2.imwrite("board.png", im)
        cv2.waitKey(0)


def draw_corners_in_image(im, corners, ids, target_width=300):
    im = im.copy()
    im, ratio = resize(im, target_width)
    corners = resize_corners(corners, ratio)
    im = cv2.aruco.drawDetectedMarkers(im, corners, ids)

    # Pose estimation
    cameraMatrix, distCoeffs = read_camera_parameters()
    rvecs, tvecs, _objPoints = cv2.aruco.estimatePoseSingleMarkers(corners, 0.05, cameraMatrix, distCoeffs)
    # Draw axis for each marker
    for i,_ in enumerate(ids):
        cv2.aruco.drawAxis(im, cameraMatrix, distCoeffs, rvecs[i], tvecs[i], 0.1)
    return im


def resize_corners(corners, ratio):
    return [ratio * c for c in corners]


def resize(im, target_width=300):
    ratio = target_width / im.shape[1]
    dim = (int(target_width), int(im.shape[0] * ratio))
    im = cv2.resize(im, dim, interpolation = cv2.INTER_LINEAR_EXACT)
    return im, ratio


def read_camera_parameters():
    camera_matrix = loadtxt(CAMERA_MATRIX, delimiter=',')
    dist_coeffs = loadtxt(DIST_COEFFS, delimiter=',')
    return camera_matrix, dist_coeffs


if __name__ == '__main__':
    filename = 'or13-equator.jpg'
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    create_board(aruco_dict, save=True)
    im = cv2.imread(filename)

    parameters =  cv2.aruco.DetectorParameters_create()
    # Refine the corners
    parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_APRILTAG
    # parameters.markerBorderBits = 1 # Important: The border width

    corners, ids, rejectedImgPoints = \
        cv2.aruco.detectMarkers(im, aruco_dict, parameters=parameters)
    print(f'{len(corners)} found markers.')
    print(f'{len(rejectedImgPoints)} rejected markers.')

    im = draw_corners_in_image(im, corners, ids, target_width=1000)

    cv2.imshow('', im)
    cv2.waitKey(0)

    # im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    # im	= cv2.adaptiveThreshold(im,
    #                             maxValue = parameters.adaptiveThreshWinSizeMax,
    #                             adaptiveMethod = cv2.ADAPTIVE_THRESH_MEAN_C, 
    #                             thresholdType = cv2.THRESH_BINARY_INV, 
    #                             blockSize = parameters.adaptiveThreshWinSizeMin, 
    #                             C = parameters.adaptiveThreshConstant)
    # cv2.imshow("", im)
    # cv2.waitKey(0)

