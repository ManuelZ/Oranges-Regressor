# Built-in imports
import argparse
from time import time

# External imports
import cv2
import numpy as np
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score

def columnize_image(im):
    N = im.shape[0] * im.shape[1]

    b, g, r = cv2.split(im)

    r = r.reshape((N, 1))
    g = g.reshape((N, 1))
    b = b.reshape((N, 1))

    X = np.concatenate([r,g,b], axis=1)
    return X, im.shape


def resize(im, target_width=100):
    ratio = target_width / im.shape[1]
    dim = (int(target_width), int(im.shape[0] * ratio))
    im = cv2.resize(im, dim, interpolation = cv2.INTER_LINEAR_EXACT)
    return im


def get_mask(column_image, resized_shape, cluster_labels, cluster_number):
    X = column_image.copy()
    cluster_labels = cluster_labels.copy()
    cluster_labels[cluster_labels == cluster_number] = 255
    cluster_labels[cluster_labels != 255] = 0
    cluster_labels = cluster_labels.astype(bool)

    X[cluster_labels] = 0
    X[~cluster_labels] = 255
    mask = X.reshape(resized_shape)[:,:,0] # Keep only one of the  layers
    return mask

if __name__ == "__main__":
    TARGET_IM_SIZE = 100

    start = time()

    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", required=True, type=str, help="Filename of input image")
    args = vars(ap.parse_args())

    im = cv2.imread(args['image'])
    im = resize(im, TARGET_IM_SIZE)
    X, resized_shape = columnize_image(im)

    ###########################################################################
    # CLUSTERING
    ###########################################################################
    model = KMeans(n_clusters=3, random_state=10)
    # model = AgglomerativeClustering(n_clusters=3, linkage='average')
    # model = AgglomerativeClustering(n_clusters=3, linkage='complete')
    # model = AgglomerativeClustering(n_clusters=3, linkage='ward')
    cluster_labels = model.fit_predict(X)
    silhouette_avg = silhouette_score(X, cluster_labels)
    print(f"The mean silhouette_score is {silhouette_avg:.2f}.")
    print(f'Found {len(np.unique(cluster_labels))} clusters')

    ###########################################################################
    # BINARIZE
    ###########################################################################
    mask0 = get_mask(X, resized_shape, cluster_labels, cluster_number=0)
    mask1 = get_mask(X, resized_shape, cluster_labels, cluster_number=1)
    mask2 = get_mask(X, resized_shape, cluster_labels, cluster_number=2)
    
    result0 = cv2.bitwise_and(im, im, mask = mask0)
    result1 = cv2.bitwise_and(im, im, mask = mask1)
    result2 = cv2.bitwise_and(im, im, mask = mask2)
    end = time()
    
    print(f'{end - start:.1f} s')
    
    cv2.imshow("Cluster 0", result0)
    cv2.imshow("Cluster 1", result1)
    cv2.imshow("Cluster 2", result2)
    cv2.waitKey(0)

