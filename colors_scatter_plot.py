# Built-in imports
import random

# External imports
import numpy as np
import cv2
from mayavi import mlab
from mayavi.mlab import quiver3d, draw

"""
Modified from:
https://stackoverflow.com/a/53405157/1253729
"""

if __name__ == '__main__':
    
    im = cv2.imread('or4-poles.jpg')
    
    # Resize image
    target_width = 300
    ratio = target_width / im.shape[1]
    dim = (int(target_width), int(im.shape[0] * ratio))
    im = cv2.resize(im, dim, interpolation = cv2.INTER_LINEAR_EXACT)
    
    N = im.shape[0] * im.shape[1] # Number of points
    b, g, r = cv2.split(im)
    
    r = r.reshape((N, 1)).astype(np.uint8)
    g = g.reshape((N, 1)).astype(np.uint8)
    b = b.reshape((N, 1)).astype(np.uint8)
    alpha = 255 * np.ones((N, 1))

    colors = np.concatenate([r, g, b, alpha], axis=1).astype(np.uint8)
    x, y, z = colors[:,0], colors[:,1], colors[:,2] 

    pts = mlab.pipeline.scalar_scatter(x, y, z)
    pts.add_attribute(colors, 'colors') # assign the colors to each point
    pts.data.point_data.set_active_scalars('colors')

    g = mlab.pipeline.glyph(pts)
    g.glyph.glyph.scale_factor = 1 # set scaling for all the points
    g.glyph.scale_mode = 'data_scaling_off' # make all the points same size

    mlab.outline()
    mlab.show()