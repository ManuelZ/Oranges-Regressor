import numpy as np
from pathlib import Path
from utils import read_camera_parameters

IMAGES_FOLDER = Path('./images')
EXT = ".jpg"

TARGET_WIDTH = 2448

DEBUG = True

REAL_H, REAL_W = 250, 160 # milimeters

CAL_MTX, DIST = read_camera_parameters()
fx = CAL_MTX[0, 0]
fy = CAL_MTX[1, 1]
FOCAL_LENGTH_PX = np.mean([fx, fy])


HSV_LOW = (0,0,0)
HSV_HIGH = (180,70,255)