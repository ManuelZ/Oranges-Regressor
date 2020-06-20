# Built-in imports
import argparse
import sys

# External imports
import cv2
import numpy as np
import cvui


ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, type=str, help="Filename of input image")
ap.add_argument("-c", "--colorspace", required=True, type=str, choices=['hsv', 'lab'], help="Colorspace to use")
args = vars(ap.parse_args())

cs = args['colorspace']
WINDOW_NAME	= 'Trackbar and columns'
width = 100 # Size of trackbars
space = 3 # inter elements space

if cs == 'hsv':
    # Vars that capture the set value in the trackbars and define the initial values
    lowCh0, highCh0 = [0], [255] # H
    lowCh1, highCh1 = [0], [110] # S
    lowCh2, highCh2 = [0], [255] # V

    # HSV ranges: (H:0-180, S:0-255, V:0-255)
    limitsCh0 = [0, 180]
    limitsCh1 = [0, 255]
    limitsCh2 = [0, 255]

elif cs == 'lab':
    # Vars that capture the set value in the trackbars and define the initial values
    lowCh0, highCh0 = [0], [255] # L
    lowCh1, highCh1 = [0], [255] # a
    lowCh2, highCh2 = [0], [135] # b

    # OpenCV CIE Lab ranges for 8 bit images
    limitsCh0 = [0, 255]
    limitsCh1 = [0, 255]
    limitsCh2 = [0, 255]

# Init cvui and tell it to use a value of 20 for cv2.waitKey()
# because we want to enable keyboard shortcut for
# all components, e.g. button with label '&Quit'.
# If cvui has a value for waitKey, it will call
# waitKey() automatically for us within cvui.update().
cvui.init(WINDOW_NAME, 20)

original_im = cv2.imread(args['image'])

# Resize so all images fit in screen
target_width = 600
ratio = target_width / original_im.shape[1]
dim = (int(target_width), int(original_im.shape[0] * ratio))
im = cv2.resize(original_im, dim, interpolation = cv2.INTER_LINEAR_EXACT)

# Will hold the transformed image and the GUI
frame = np.zeros(im.shape, np.uint8)

while True:
    ###########################################################################
    # BLUR
    ###########################################################################
    frameBGR = cv2.GaussianBlur(im, (7, 7), 0)

    ###########################################################################
    # COLORSPACE
    ###########################################################################
    if cs == 'hsv':
        transformed = cv2.cvtColor(frameBGR, cv2.COLOR_BGR2HSV)
    elif cs == 'lab':
        transformed = cv2.cvtColor(frameBGR, cv2.COLOR_BGR2LAB)
    
    # Colorspace values  define a color range
    colorLow = np.array([lowCh0, lowCh1, lowCh2])
    colorHigh = np.array([highCh0, highCh1, highCh2])
    mask = cv2.inRange(transformed, colorLow, colorHigh)

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
    options = cvui.TRACKBAR_DISCRETE | cvui.TRACKBAR_HIDE_SEGMENT_LABELS
    
    cvui.beginColumn(frame, 10, 40, -1, -1, 6)

    if cs == 'hsv':
        cvui.text('Hue low')
    elif cs == 'lab':
        cvui.text('L low')
    cvui.trackbar(width, lowCh0, limitsCh0[0], limitsCh0[1], 10, '%.0Lf', options, 1)
    cvui.space(space)

    if cs == 'hsv':
        cvui.text('Sat low')
    elif cs == 'lab':
        cvui.text('a* low')
    cvui.trackbar(width, lowCh1, limitsCh1[0], limitsCh1[1], 10, '%.0Lf', options, 1)
    cvui.space(space)

    if cs == 'hsv':
        cvui.text('Val low')
    elif cs == 'lab':
        cvui.text('b* low')
    cvui.trackbar(width, lowCh2, limitsCh2[0], limitsCh2[1], 10, '%.0Lf', options, 1)
    cvui.space(space)

    cvui.endColumn()

    ###########################################################################
    # SECOND COLUMN
    ###########################################################################
    cvui.beginColumn(frame, 110, 40, -1, -1, 6)
    if cs == 'hsv':
        cvui.text('Hue high')
    elif cs == 'lab':
        cvui.text('L high')
    cvui.trackbar(width, highCh0, limitsCh0[0], limitsCh0[1], 10, '%.0Lf', options, 1)
    cvui.space(space)

    if cs == 'hsv':
        cvui.text('Sat high')
    elif cs == 'lab':
        cvui.text('a* high')
    cvui.trackbar(width, highCh1, limitsCh1[0], limitsCh1[1], 10, '%.0Lf', options, 1)
    cvui.space(space)

    if cs == 'hsv':
        cvui.text('Val high')
    elif cs == 'lab':
        cvui.text('b* high')
    cvui.trackbar(width, highCh2, limitsCh2[0], limitsCh2[1], 10, '%.0Lf', options, 1)
    cvui.space(space)

    cvui.endColumn()

    # Check if ESC key was pressed
    if cv2.waitKey(20) == 27:
        break

    # Since cvui.init() received a param regarding waitKey,
    # there is no need to call cv.waitKey() anymore. cvui.update()
    # will do it automatically.
    cvui.update()

    cv2.imshow(WINDOW_NAME, frame)

