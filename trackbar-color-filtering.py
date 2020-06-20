"""
MODIFIED FROM
https://www.bluetin.io/opencv/opencv-color-detection-filtering-python/
"""
import argparse
import sys
import cv2
import numpy as np
import cvui

# TODO: Try the CIE L*a*b* colorspace. cv2.COLOR_BGR2LAB
# This outputs 0≤L≤100, −127≤a≤127, −127≤b≤127 .
# HSV: OpenCV uses HSV ranges between (H:0-180, S:0-255, V:0-255)

WINDOW_NAME	= 'Trackbar and columns'

lowHue = [0]
lowSat = [0]
lowVal = [0]
highHue = [255]
highSat = [110]
highVal = [255]

# Size of trackbars
width = 100

# Init cvui and tell it to use a value of 20 for cv2.waitKey()
# because we want to enable keyboard shortcut for
# all components, e.g. button with label '&Quit'.
# If cvui has a value for waitKey, it will call
# waitKey() automatically for us within cvui.update().
cvui.init(WINDOW_NAME, 20)


ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, type=str, help="Filename of input image")
args = vars(ap.parse_args())

original_im = cv2.imread(args['image'])

# Resize so all images fit in screen
target_width = 600
ratio = target_width / original_im.shape[1]
dim = (int(target_width), int(original_im.shape[0] * ratio))
im = cv2.resize(original_im, dim, interpolation = cv2.INTER_LINEAR_EXACT)

frame = np.zeros(im.shape, np.uint8)

while True:
    ###########################################################################
    # BLUR
    ###########################################################################
    # Blur methods available, comment or uncomment to try different blur methods.
    frameBGR = cv2.GaussianBlur(im, (7, 7), 0)

    ###########################################################################
    # HSV COLORSPACE
    ###########################################################################
    hsv = cv2.cvtColor(frameBGR, cv2.COLOR_BGR2HSV)
    
    # HSV values to define a colour range.
    colorLow = np.array([lowHue,lowSat,lowVal])
    colorHigh = np.array([highHue,highSat,highVal])
    mask = cv2.inRange(hsv, colorLow, colorHigh)
    h, s, v = cv2.split(hsv)

    ###########################################################################
    # MORPHOLOGY
    ###########################################################################
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    
    ###########################################################################
    # RESULTS
    ###########################################################################
    # Invert mask
    mask = cv2.bitwise_not(mask)

    # Put mask over top of the original image.
    result = cv2.bitwise_and(im, im, mask = mask)

    frame[:] = result[:]
 
    ###########################################################################
    # GUI
    ###########################################################################
    # Render the settings window to house the checkbox and the trackbars below
    cvui.window(frame, 10, 10, 200, 250, 'Settings')
    
    cvui.beginColumn(frame, 10, 40, -1, -1, 6)
    
    options = cvui.TRACKBAR_DISCRETE | cvui.TRACKBAR_HIDE_SEGMENT_LABELS

    cvui.text('Hue low')
    cvui.trackbar(width, lowHue, 0, 180, 10, '%.0Lf', options, 1)
    cvui.space(5)

    cvui.text('Sat low')
    cvui.trackbar(width, lowSat, 0, 255, 10, '%.0Lf', options, 1)
    cvui.space(5)

    cvui.text('Value low')
    cvui.trackbar(width, lowVal, 0, 255, 10, '%.0Lf', options, 1)
    cvui.space(5)

    cvui.endColumn()

    ###########################################################################
    # SECOND COLUMN
    ###########################################################################
    cvui.beginColumn(frame, 110, 40, -1, -1, 6)
    cvui.text('Hue high')
    cvui.trackbar(width, highHue, 0, 180, 10, '%.0Lf', options, 1)
    cvui.space(5)

    cvui.text('Sat high')
    cvui.trackbar(width, highSat, 0, 255, 10, '%.0Lf', options, 1)
    cvui.space(5)

    cvui.text('Value high')
    cvui.trackbar(width, highVal, 0, 255, 10, '%.0Lf', options, 1)
    cvui.space(5)

    cvui.endColumn()

    # Check if ESC key was pressed
    if cv2.waitKey(20) == 27:
        break

    # Since cvui.init() received a param regarding waitKey,
    # there is no need to call cv.waitKey() anymore. cvui.update()
    # will do it automatically.
    cvui.update()

    cv2.imshow(WINDOW_NAME, frame)

