# Built-in imports
import sys
import logging
from itertools import combinations
import argparse

# External imports
import cv2
import numpy as np
from numpy.lib import recfunctions as rfn
from imutils.perspective import four_point_transform
from imutils.perspective import order_points
from sklearn.neighbors import NearestNeighbors
from my_opencv_utils import Processer
from scipy.spatial.distance import euclidean

# Local imports
from setup_logger import logger
from config import CAL_MTX
from config import DIST
from config import TARGET_WIDTH
from config import FOCAL_LENGTH_PX
from config import REAL_L, REAL_W
from config import DEBUG
from config import EXT
from config import HSV_LOW, HSV_HIGH
from exceptions import BlobError

# Supress numpy scientific notation
np.set_printoptions(precision=2, suppress=True, threshold=5)

###############################################################################


def find_squares(im):
    """
    Return a list of keypoints with the centroids of the found squares.
    """
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(im, connectivity=8)
    labels = labels.astype(np.uint8)
    
    new_stats = []
    for i in range(num_labels):
        blob = labels.copy()
        blob[np.where(blob == i)] = 255
        blob[np.where(blob != 255)] = 0
        try:
            area, perimeter, circularity, bbox_area = get_blob_properties(blob)
            new_stats.append((area, perimeter, circularity, bbox_area, centroids[i][0], centroids[i][1]))
        except BlobError:
            pass
    
    structured = np.array(new_stats, dtype=({ 'names' : ['area', 'perimeter', 'circularity', 'bbox_area', 'cx', 'cy'],
                                              'formats' : ['f4', 'f4', 'f4', 'f4', 'f4', 'f4']
                                            }))
    unstructured = rfn.structured_to_unstructured(structured[['area', 'perimeter', 'circularity', 'bbox_area']])

    if unstructured.shape[0] >= 4:
        nn = NearestNeighbors(n_neighbors=4, algorithm='ball_tree')
        model = nn.fit(unstructured)
        indices = model.kneighbors(unstructured, return_distance=False)
        # remove duplicate groups
        results = np.array([list(item) for item in set(frozenset(item) for item in indices)])
        
        # Select the best group by minimizing the sum of std
        min_std = np.inf
        selected = None
        for group in results:
            d = np.std(unstructured[group], axis=0).sum()
            if d < min_std:
                min_std = d
                selected = group
        logger.debug(f'Selected group is: {selected}')
    else:
        logger.error("Not enough square blobs found")
        sys.exit()

    selected = structured[selected]
    keypoints = [cv2.KeyPoint(x=row['cx'], y=row['cy'], _size=np.sqrt(row['area'])) for row in selected]
    return keypoints


def find_orange(im):

    # TODO: use Canny or change the HSV range

    logger.info(f'Searching orange...')
    num_labels, labels, stats, centroids = \
        cv2.connectedComponentsWithStats(im, connectivity=8)

    labels = labels.astype(np.uint8)
    areas = stats[:,4]
    
    # Find the blob with the max circularity
    max_circularity = -np.inf
    selected_idx = None
    orange_blob = None
    for i in range(num_labels):
        blob = labels.copy()
        blob[np.where(blob == i)] = 255
        blob[np.where(blob != 255)] = 0
        try:
            area, perimeter, circularity, bbox_area = get_blob_properties(blob)
            if circularity > max_circularity:
                max_circularity = circularity
                selected_idx = i
                orange_blob = blob
        except BlobError:
            pass
    
    _, contours, _ = cv2.findContours(orange_blob, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    ellipse_center, ellipse_size, ellipse_angle = cv2.fitEllipse(contours[0])
    empty_canvas = np.zeros(orange_blob.shape)

    orange_centroid = centroids[selected_idx]
    orange_min_radius_px = np.min(ellipse_size) / 2
    keypoints = [cv2.KeyPoint(x=orange_centroid[0], 
                              y=orange_centroid[1], 
                              _size=2*orange_min_radius_px)]
    return keypoints, float(2*orange_min_radius_px)


def calc_orange_real_diam(ref_mm, ref_px, orange_diam_px, focal_length_px):
    orange_diam_mm = ref_mm * (orange_diam_px / ref_px ) * ( (2 * focal_length_px) / (2 * focal_length_px + orange_diam_px))
    return orange_diam_mm


def calculate_orange_volume(equator_diameter, poles_diameter=None):
    
    r = equator_diameter / 2

    if poles_diameter is not None:
        # https://en.wikipedia.org/wiki/Spheroid#Volume
        a = poles_diameter / 2
        vol_cubic_mm = (4 / 3) * np.pi * np.power(r, 2) * a
    else:
        # Sphere volume
        vol_cubic_mm = (4 / 3) * np.pi * np.power(r, 3)
    
    vol_ml = vol_cubic_mm * 1e-3
    return vol_ml


def get_reference_length_and_width(centroids):
    # order coordinates clockwise
    tl, tr, br, bl = order_points(centroids)
    w = euclidean(br, bl)
    l = euclidean(tl, bl)
    return l, w


def get_blob_properties(blob):
    _,contours,hierarchy = \
        cv2.findContours(blob, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        raise BlobError
    cnt = contours[0]
    area = cv2.contourArea(cnt)
    if area < 10: raise BlobError
    perimeter = cv2.arcLength(cnt, closed=True)
    circularity = (4 * np.pi * area) / np.power(perimeter, 2)
    x,y,w,h = cv2.boundingRect(cnt)
    bbox_area = w * h
    return area, perimeter, circularity, bbox_area


def drawKeyPts(im, keypts, color, th):
    for kp in keypts:
        x = np.int(kp.pt[0])
        y = np.int(kp.pt[1])
        size = np.int(kp.size / 2)
        cv2.circle(im, (x,y), size, color, thickness=th, lineType=8, shift=0) 
    return im 


def draw_line_with_long(im, pt1, pt2, mm_per_px):
    pt1, pt2 = tuple(pt1), tuple(pt2)
    cv2.line(im, pt1, pt2, color = (0,0,0), thickness = 2, lineType = cv2.LINE_8)
    x = int(pt1[0] + (pt2[0] - pt1[0]) / 2)
    y = int(pt1[1] + (pt2[1] - pt1[1]) / 2) - 10
    cv2.putText(im,
                text = f"{euclidean(pt1,pt2):.0f}px", 
                org = (x, y), 
                fontFace = cv2.FONT_HERSHEY_SIMPLEX, 
                fontScale = 1.5, 
                color = (0,0,255), 
                thickness = 5)
    cv2.putText(im,
                text = f"{mm_per_px * euclidean(pt1,pt2):.0f}mm", 
                org = (x, y + 50), 
                fontFace = cv2.FONT_HERSHEY_SIMPLEX, 
                fontScale = 1.5, 
                color = (0,0,255), 
                thickness = 5)


def estimate_volume(equator_filename, poles_filename):
    """
    Args:
        equator_filename:
        poles_filename: 
    
    Returns:
        
    """
    diameters = []
    for fn in equator_filename, poles_filename:
        processer = \
            Processer(f'{fn}{EXT}', TARGET_WIDTH, CAL_MTX, DIST, debug=DEBUG)

        #######################################################################
        # Find Orange
        #######################################################################
        (
            processer
                .resize()
                .blur(3)
                .show("Orange")
                .undistort()
                .show("Orange")
                .to_hsv()
                .show("Orange")
                .filter_by_hsv_color(lowHSV=HSV_LOW, highHSV=HSV_HIGH)
                .show("Orange")
                .negate()
                .show("Orange")
                .open(size=10, iterations=5, element='circle')
                .close(size=10, iterations=5, element='circle')
                .show("Orange")
        )
        im = processer.get_processed_image()
        orange_kpts, orange_diam_px = find_orange(im)
        orange_centroids = np.array([kp.pt for kp in orange_kpts])

        #######################################################################
        # Find squares
        #######################################################################
        (
            processer
                .reset()
                .resize()
                .blur(3)
                #.show('Squares')
                .undistort()
                #.show('Squares')
                .to_gray()
                #.show('Squares')
                .binarize(method='otsu')
                #.show('Squares')
                .open(size=10, iterations=5, element='rectangle')
                .close(size=5, iterations=5, element='rectangle')
                #.show('Squares')
        )
        im = processer.get_processed_image()
        squares_kpts = find_squares(im)
        squares_centroids = np.array([kp.pt for kp in squares_kpts])

        #
        # Length and Width
        #
        l,w = get_reference_length_and_width(squares_centroids)
        logger.debug(f'Squares length: {l:.1f} px; Squares width: {w:.1f} px')


        #######################################################################
        # Transform pixels to mm
        #######################################################################
        mm_per_px1 = REAL_L / l
        mm_per_px2 = REAL_W / w
        logger.debug(f'Reference [mm]: {REAL_L:.1f} mm')
        logger.debug(f'Reference [px]: {l:.1f} px')
        logger.debug(f'Image orange diameter: {orange_diam_px:.1f} px')
        logger.debug(f'Focal length: {FOCAL_LENGTH_PX:.1f} px')

        orange_diameter = calc_orange_real_diam(ref_mm=REAL_L, 
                                                ref_px=l, 
                                                orange_diam_px=orange_diam_px, 
                                                focal_length_px=FOCAL_LENGTH_PX)

        logger.info(f'Estimated real orange diameter: {orange_diameter:.1f} mm')

        #######################################################################
        # Draw
        #######################################################################
        im = (
                processer
                    .reset()
                    .resize()
                    .undistort()
                    .get_processed_image()
            )

        # Mark the squares and orange
        im_with_keypoints = drawKeyPts(im, orange_kpts, (0,0,255), 10)
        im_with_keypoints = \
            cv2.drawKeypoints(im_with_keypoints, squares_kpts, np.array([]), 
                              (0,255,0), 
                              cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS) 

        # Draw squares locations
        for kp in squares_kpts:
            cv2.putText(im_with_keypoints,
                        text = f"{kp.pt[0]:.0f}, {kp.pt[1]:.0f}", 
                        org = (int(kp.pt[0] - 30), int(kp.pt[1] - 30)), 
                        fontFace = cv2.FONT_HERSHEY_SIMPLEX, 
                        fontScale = 1.5, 
                        color = (0,0,255), 
                        thickness = 5)

        # Draw lines between squares
        tl, tr, br, bl = order_points(squares_centroids)
        draw_line_with_long(im_with_keypoints, tl, tr, mm_per_px1)
        draw_line_with_long(im_with_keypoints, br, tr, mm_per_px1)
        draw_line_with_long(im_with_keypoints, bl, br, mm_per_px1)
        draw_line_with_long(im_with_keypoints, bl, tl, mm_per_px1)

        # Draw orange diameter
        orange_diam_text_x = int(orange_centroids[0][0])
        orange_diam_text_y = int(orange_centroids[0][1] - (orange_diam_px / 2) - 10)
        cv2.putText(im_with_keypoints,
                    text = f"{orange_diameter:.1f} mm", 
                    org = (orange_diam_text_x, orange_diam_text_y), 
                    fontFace = cv2.FONT_HERSHEY_SIMPLEX, 
                    fontScale = 3.5, 
                    color = (0,0,255), 
                    thickness = 5)

        cv2.imwrite(f'{fn}-out{EXT}', im_with_keypoints)
        diameters.append(orange_diameter)
        logger.info(f'Finished with one orange.')
    
    ###########################################################################
    # FINAL CALC
    ###########################################################################
    orange_volume = calculate_orange_volume(diameters[0], diameters[1])
    logger.info(f'Finished with orange {equator_filename}, {poles_filename}')
    logger.info(f'Estimated real orange volume: {orange_volume:.1f} ml\n')
    return orange_volume


if __name__ == '__main__':
    # ap = argparse.ArgumentParser()
    # ap.add_argument("-o", "--orange", required=True, type=int, help="Orange number")
    # args = vars(ap.parse_args())
    # orange_num = args['orange']

    for orange_num in range(7, 13):
        equator_filename, poles_filename = f"or{orange_num}-equator", f"or{orange_num}-poles"
        est_vol = estimate_volume(equator_filename, poles_filename)

    # TODO: transform to some color space and do histogram equalization as in
    # the psid lab