# Built-in imports
import argparse
from time import time

# External imports
import cv2
import numpy as np
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score

"""
Summary
An image in the HSV colorspace is expressed as a three column matrix and passed
to a KMeans clusterer.

Conclusion
The clustering takes way too much time. A trick can be used of resizing the
image to a very reduced scale (i.e. 100 px width) so that the clustering doesn't
take so much time. Then the obtained mask can be resized back to the original 
image size. But this trick gives very coarse results because of the upsampling.


"""



def columnize_image(im, num_layers=3):
    N = im.shape[0] * im.shape[1]
    original_shape = (im.shape[0], im.shape[1], num_layers)
        
    h,s,v = cv2.split(im)
    h = h.reshape((N, 1))
    s = s.reshape((N, 1))
    v = v.reshape((N, 1))

    if num_layers == 2:
        X = np.concatenate([h,s], axis=1)
    else:
        X = np.concatenate([h,s,v], axis=1)
    
    return X, original_shape


def resize(im, target_width=100, target_height=None):
    """
    Passing targe_height means forcing the final size, better use only 
    target_width to automatically keep the original W/L ratio, unless you know 
    what you are doing.
    """
    if target_height:
        dim = (int(target_width), int(target_height))
    else:
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
    mask = X.reshape(resized_shape)[:,:,0] # Keep only one of the layers
    return mask

if __name__ == "__main__":
    TARGET_IM_SIZE = 200

    start = time()

    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", required=True, type=str, help="Filename of input image")
    args = vars(ap.parse_args())

    original_im = cv2.imread(args['image'])
    hsv = cv2.cvtColor(original_im, cv2.COLOR_BGR2HSV)
    im = resize(hsv, TARGET_IM_SIZE)
    X, resized_shape = columnize_image(im, num_layers=2)

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
    masks = []
    results = []
    for i in np.unique(cluster_labels):
        mask = get_mask(X, resized_shape, cluster_labels, cluster_number=i)
        original_im_resized = resize(original_im, TARGET_IM_SIZE)
        result = cv2.bitwise_and(original_im_resized, original_im_resized, mask = mask)
        cv2.imshow(f"Cluster {i}", result)
    
    end = time()
    print(f'{end - start:.1f} s')
    cv2.waitKey(0)

    # Use the mask obtained in the small image to segment the big image
    # im300 = resize(original_im, 300)
    # h,w,_ = im300.shape
    # mask0_300 = resize(mask0, w, h)
    # mask1_300 = resize(mask1, w, h)
    # mask2_300 = resize(mask2, w, h)

    # result0 = cv2.bitwise_and(im300, im300, mask = mask0_300)
    # result1 = cv2.bitwise_and(im300, im300, mask = mask1_300)
    # result2 = cv2.bitwise_and(im300, im300, mask = mask2_300)
    

