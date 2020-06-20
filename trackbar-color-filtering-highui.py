"""
MODIFIED FROM
https://www.bluetin.io/opencv/opencv-color-detection-filtering-python/
"""
import argparse
import sys
import cv2
import numpy as np

# HSV: OpenCV uses HSV ranges between (H:0-180, S:0-255, V:0-255)


def callback(value=None):
    ###########################################################################
    # EXTRACT VALUE LIMITS
    ###########################################################################
    # Get the values from the GUI sliders
    lowCh0 = cv2.getTrackbarPos('lowCh0', 'output')
    lowCh1 = cv2.getTrackbarPos('lowCh1', 'output')
    lowCh2 = cv2.getTrackbarPos('lowCh2', 'output')
    highCh0 = cv2.getTrackbarPos('highCh0', 'output')
    highCh1 = cv2.getTrackbarPos('highCh1', 'output')
    highCh2 = cv2.getTrackbarPos('highCh2', 'output')


    ###########################################################################
    # ORIGINAL IMAGE
    ###########################################################################
    # cv2.imshow('frame', frame)


    ###########################################################################
    # BLUR
    ###########################################################################
    frameBGR = cv2.GaussianBlur(frame, (7, 7), 0)
    # cv2.imshow('blurred', frameBGR)


    ###########################################################################
    # COLORSPACE
    ###########################################################################
    if args['colorspace'] == 'hsv':
        transformed = cv2.cvtColor(frameBGR, cv2.COLOR_BGR2HSV)
    if args['colorspace'] == 'lab':
        transformed = cv2.cvtColor(frameBGR, cv2.COLOR_BGR2LAB)

    # Values to define a colour range
    colorLow = np.array([lowCh0, lowCh1, lowCh2])
    colorHigh = np.array([highCh0, highCh1, highCh2])
    mask = cv2.inRange(transformed, colorLow, colorHigh)
    # cv2.imshow('mask-color-filter', mask)

    ###########################################################################
    # MORPHOLOGY
    ###########################################################################
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # cv2.imshow('mask-morphological', mask)


    ###########################################################################
    # RESULTS
    ###########################################################################

    # Invert mask
    mask = cv2.bitwise_not(mask)

    # Put mask over top of the original image.
    result = cv2.bitwise_and(frame, frame, mask = mask)

    # Show final output image
    cv2.imshow('output', result)



ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, type=str, help="Filename of input image")
ap.add_argument("-c", "--colorspace", required=True, type=str, choices=['hsv', 'lab'], help="Colorspace to use")
args = vars(ap.parse_args())

frame = cv2.imread(args['image'])

if args['colorspace'] == 'hsv':
    # HSV ranges: (H:0-180, S:0-255, V:0-255)
    limitsCh0 = [0, 180]
    limitsCh1 = [0, 255]
    limitsCh2 = [0, 255]

    # Initial values
    ch0 = (0, 255) # H
    ch1 = (0, 110) # S
    ch2 = (0, 255) # V

elif args['colorspace'] == 'lab':
    # OpenCV CIE Lab ranges for 8 bit images
    limitsCh0 = [0, 255] # L*
    limitsCh1 = [0, 255] # a*
    limitsCh2 = [0, 255] # b*

    # Initial values
    ch0 = (0, 255) # L*
    ch1 = (0, 255) # a*
    ch2 = (0, 135) # b*

cv2.namedWindow('output', cv2.WINDOW_AUTOSIZE)

# Lower range colour sliders
cv2.createTrackbar('lowCh0', 'output', ch0[0], 255, callback)
cv2.createTrackbar('lowCh1', 'output', ch1[0], 255, callback)
cv2.createTrackbar('lowCh2', 'output', ch2[0], 255, callback)

# Higher range colour sliders
cv2.createTrackbar('highCh0', 'output', ch0[1], 255, callback)
cv2.createTrackbar('highCh1', 'output', ch1[1], 255, callback)
cv2.createTrackbar('highCh2', 'output', ch2[1], 255, callback)

# Resize so all images fit in screen
target_width = 300
ratio = target_width / frame.shape[1]
dim = (int(target_width), int(frame.shape[0] * ratio))
frame = cv2.resize(frame, dim, interpolation = cv2.INTER_LINEAR_EXACT)

callback()

while True:
    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        cv2.destroyAllWindows()
        break 