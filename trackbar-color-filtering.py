"""
MODIFIED FROM
https://www.bluetin.io/opencv/opencv-color-detection-filtering-python/
"""
import argparse
import sys
import cv2
import numpy as np

# TODO: Try the CIE L*a*b* colorspace. cv2.COLOR_BGR2LAB
# This outputs 0≤L≤100, −127≤a≤127, −127≤b≤127 .
# HSV: OpenCV uses HSV ranges between (H:0-180, S:0-255, V:0-255)


def callback(value=None):
    # Get HSV values from the GUI sliders.
    lowHue = cv2.getTrackbarPos('lowHue', 'colorTest')
    lowSat = cv2.getTrackbarPos('lowSat', 'colorTest')
    lowVal = cv2.getTrackbarPos('lowVal', 'colorTest')
    highHue = cv2.getTrackbarPos('highHue', 'colorTest')
    highSat = cv2.getTrackbarPos('highSat', 'colorTest')
    highVal = cv2.getTrackbarPos('highVal', 'colorTest')
 

    ###########################################################################
    # ORIGINAL IMAGE
    ###########################################################################
    cv2.imshow('frame', frame)
    

    ###########################################################################
    # BLUR
    ###########################################################################
    # Blur methods available, comment or uncomment to try different blur methods.
    frameBGR = cv2.GaussianBlur(frame, (7, 7), 0)
    #frameBGR = cv2.medianBlur(frameBGR, 7)
    #frameBGR = cv2.bilateralFilter(frameBGR, 15 ,75, 75)
    # kernel = np.ones((15, 15), np.float32)/255
    # frameBGR = cv2.filter2D(frameBGR, -1, kernel)	

    cv2.imshow('blurred', frameBGR)
	

    ###########################################################################
    # HSV COLORSPACE
    ###########################################################################
    # HSV (Hue, Saturation, Value).
    # Convert the frame to HSV colour model.
    hsv = cv2.cvtColor(frameBGR, cv2.COLOR_BGR2HSV)
    
    # HSV values to define a colour range.
    colorLow = np.array([lowHue,lowSat,lowVal])
    colorHigh = np.array([highHue,highSat,highVal])
    mask = cv2.inRange(hsv, colorLow, colorHigh)
    # Show the first mask
    cv2.imshow('mask-color-filter', mask)

    h, s, v = cv2.split(hsv)

    ###########################################################################
    # MORPHOLOGY
    ###########################################################################
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
 
    # Show morphological transformation mask
    cv2.imshow('mask-morphological', mask)
    
    
    ###########################################################################
    # RESULTS
    ###########################################################################

    # Invert mask
    mask = cv2.bitwise_not(mask)

    # Put mask over top of the original image.
    result = cv2.bitwise_and(frame, frame, mask = mask)
 
    # Show final output image
    cv2.imshow('colorTest', result)

    

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, type=str, help="Filename of input image")
args = vars(ap.parse_args())

frame = cv2.imread(args['image'])

hue = (0, 255)
sat = (0, 110)
val = (0, 255)

cv2.namedWindow('colorTest', cv2.WINDOW_AUTOSIZE)

# Lower range colour sliders.
cv2.createTrackbar('lowHue', 'colorTest', hue[0], 255, callback)
cv2.createTrackbar('lowSat', 'colorTest', sat[0], 255, callback)
cv2.createTrackbar('lowVal', 'colorTest', val[0], 255, callback)

# Higher range colour sliders.
cv2.createTrackbar('highHue', 'colorTest', hue[1], 255, callback)
cv2.createTrackbar('highSat', 'colorTest', sat[1], 255, callback)
cv2.createTrackbar('highVal', 'colorTest', val[1], 255, callback)
 
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