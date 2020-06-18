import numpy as np

EXT = ".jpg"

TARGET_WIDTH = 2448

DEBUG = False

REAL_L, REAL_W = 240,150 # milimeters

# Camera parameters calculated with Matlab
fx, fy = 2.9496e+03, 2.9535e+03 
# Camera parameters calculated with Opencv
# fx, fy = 2.59010722e+03, 2.58287227e+03 

FOCAL_LENGTH_PX = np.mean([fx, fy])

CAL_MTX = np.array([[fx, 0, 1.64003469e+03],
                    [0, fy, 1.20986771e+03],
                    [0, 0, 1]])

DIST = np.array([[9.81063604e-02, -3.04860599e-01, 3.62797741e-04, 3.14197889e-04, 3.30625745e-01]])