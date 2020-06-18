# Built-in imports
import sys
import logging
from itertools import combinations

# External imports
import cv2
import numpy as np
from numpy.lib import recfunctions as rfn
from scipy.spatial import distance as dist
from imutils.perspective import four_point_transform
from imutils.perspective import order_points
from sklearn.neighbors import NearestNeighbors
from my_opencv_utils import Processer
from scipy.spatial.distance import euclidean

# Local imports
from setup_logger import logger

# Supress numpy scientific notation
np.set_printoptions(precision=2, suppress=True, threshold=5)

###############################################################################
extension = ".jpg"

#filename = "or3-equator"
filename = "or3-poles"

# target_width = 1000
target_width = 2448

DEBUG = False

REAL_L, REAL_W = 240,150 # milimeters

# Matlab
fx, fy = 2.9496e+03, 2.9535e+03 
FOCAL_LENGTH_PX = np.mean([fx,fy])
# Opencv
#fx, fy = 2.59010722e+03, 2.58287227e+03 

CAL_MTX = np.array([[fx, 0, 1.64003469e+03],
                    [0, fy, 1.20986771e+03],
                    [0, 0, 1]])

DIST = np.array([[9.81063604e-02, -3.04860599e-01, 3.62797741e-04, 3.14197889e-04, 3.30625745e-01]])


class BlobError(Exception):
    """ Raised when a blob is a point"""
    def __str__(self):
        return "One of the found blobs is a point or almost one."
  

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
        
        # Select the best group by minimizing the std
        k = np.inf
        selected = None
        for group in results:
            d = np.std(unstructured[group], axis=0).sum()
            if d < k:
                k = d
                selected = group
        logger.debug(f'Selected group is: {selected}')
    else:
        logger.error("Not enough square blobs found")
        sys.exit()

    selected = structured[selected]
    keypoints = [cv2.KeyPoint(x=row['cx'], y=row['cy'], _size=np.sqrt(row['area'])) for row in selected]
    return keypoints


def find_orange(im):
    logger.info(f'Searching orange...')
    num_labels, labels, stats, centroids = \
        cv2.connectedComponentsWithStats(im, connectivity=8)

    labels = labels.astype(np.uint8)
    areas = stats[:,4]
    
    max_circularity = -np.inf
    selected_idx = None
    for i in range(num_labels):
        blob = labels.copy()
        blob[np.where(blob == i)] = 255
        blob[np.where(blob != 255)] = 0
        try:
            area, perimeter, circularity, bbox_area = get_blob_properties(blob)
            if circularity > max_circularity:
                max_circularity = circularity
                selected_idx = i
        except BlobError:
            pass

    orange_centroid = centroids[selected_idx]
    orange_area_px = areas[selected_idx]
    logger.info(f'Orange area (px): {orange_area_px} px')
    orange_radius_px = np.sqrt(orange_area_px / np.pi)
    keypoints = [cv2.KeyPoint(x=orange_centroid[0], 
                              y=orange_centroid[1], 
                              _size=2*orange_radius_px)]
    return keypoints, float(2*orange_radius_px)


def calculate_orange_real_diameter(ref_mm, ref_px, orange_diam_px, focal_length_px):
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


def draw_line_with_long(im, pt1, pt2):
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
                text = f"{mm_per_px1 * euclidean(pt1,pt2):.0f}mm", 
                org = (x, y + 50), 
                fontFace = cv2.FONT_HERSHEY_SIMPLEX, 
                fontScale = 1.5, 
                color = (0,0,255), 
                thickness = 5)


processer = Processer(f'{filename}{extension}', target_width, CAL_MTX, DIST, debug=DEBUG)

###############################################################################
# Find Orange
###############################################################################
(
    processer
        .resize()
        .blur(3)
        #.show("Orange")
        .undistort()
        #.show("Orange")
        #.extract_page()
        #.show("Orange")
        .to_hsv()
        #.show("Orange")
        .filter_by_hsv_color(lowHSV=[0, 0, 62], highHSV=[148, 68, 255])
        #.show("Orange")
        .negate()
        #.show("Orange")
        .open(size=10, iterations=2, element='circle')
        .close(size=10, iterations=2, element='circle')
        #.show("Orange")
)
im = processer.get_processed_image()
orange_kpts, orange_diam_px = find_orange(im)
orange_centroids = np.array([kp.pt for kp in orange_kpts])

###############################################################################
# Find squares
###############################################################################
(
    processer
        .reset()
        .resize()
        .blur(3)
        #.show('Squares')
        .undistort()
        #.show('Squares')
        #.extract_page()
        #.show('Squares')
        .to_gray()
        #.show('Squares')
        .binarize(method='otsu')
        #.show('Squares')
        .open(size=10, iterations=2, element='rectangle')
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
logger.info(f'length: {l:.1f} px; width: {w:.1f} px')


###############################################################################
# Transform pixels to mm
###############################################################################
mm_per_px1 = REAL_L / l
mm_per_px2 = REAL_W / w
logger.debug(f'Reference [mm]: {REAL_L:.1f} mm')
logger.debug(f'Reference [px]: {l:.1f} px')
logger.debug(f'Image orange diameter: {orange_diam_px:.1f} px')
logger.debug(f'Focal length: {FOCAL_LENGTH_PX:.1f} px')
# logger.debug(f'{mm_per_px1:.4f} mm per px.')
# logger.debug(f'{mm_per_px2:.4f} mm per px.')
# logger.debug(f'Less than {1 - mm_per_px1/mm_per_px2:.2%} of difference between them.')

orange_diameter = calculate_orange_real_diameter(ref_mm=REAL_L, ref_px=l, orange_diam_px=orange_diam_px, focal_length_px=FOCAL_LENGTH_PX)
orange_volume = calculate_orange_volume(orange_diameter)

logger.info(f'Estimated real orange diameter: {orange_diameter:.1f} mm')
logger.info(f'Estimated real orange volume: {orange_volume:.1f} ml')


###############################################################################
# Draw
###############################################################################
im = (processer
          .reset()
          .resize()
          .undistort()
          #.extract_page()
          .get_processed_image()
     )

# Mark the squares and orange
im_with_keypoints = drawKeyPts(im, orange_kpts, (0,0,255), 10)
im_with_keypoints = cv2.drawKeypoints(im_with_keypoints, squares_kpts, np.array([]), (0,255,0), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS) 

 
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
draw_line_with_long(im_with_keypoints, tl, tr)
draw_line_with_long(im_with_keypoints, br, tr)
draw_line_with_long(im_with_keypoints, bl, br)
draw_line_with_long(im_with_keypoints, bl, tl)
    

# Draw orange diameter
cv2.putText(im_with_keypoints,
            text = f"{orange_diameter:.1f} mm", 
            org = (int(orange_centroids[0][0]), int(orange_centroids[0][1] - (orange_diam_px / 2) - 10 )), 
            fontFace = cv2.FONT_HERSHEY_SIMPLEX, 
            fontScale = 3.5, 
            color = (0,0,255), 
            thickness = 5)

cv2.imwrite(f'{filename}-out{extension}', im_with_keypoints)
#cv2.imshow("Squares and orange", im_with_keypoints)
#cv2.waitKey(0)
