import numpy as np
from tvtk.api import tvtk
from mayavi.scripts import mayavi2
import vispy.scene
import random
from mayavi import mlab
from mayavi.mlab import quiver3d, draw
import cv2

"""
Modified from:
https://github.com/enthought/mayavi/issues/92
"""

mlab.options.backend = 'envisage'

if __name__ == '__main__':
    
    im = cv2.imread('or4-poles.jpg')
    
    # Resize image
    target_width = 300
    ratio = target_width / im.shape[1]
    dim = (int(target_width), int(im.shape[0] * ratio))
    im = cv2.resize(im, dim, interpolation = cv2.INTER_LINEAR_EXACT)
    
    N = im.shape[0] * im.shape[1] # Number of points
    ones = np.ones(N)
    scalars = np.arange(N) # Key point: set an integer for each point
    
    b, g, r = cv2.split(im)
    alpha = (255 * np.ones((r.size, 1))).astype(np.uint8)
    r = r.reshape((r.size, 1)).astype(np.uint8)
    g = g.reshape((g.size, 1)).astype(np.uint8)
    b = b.reshape((b.size, 1)).astype(np.uint8)
    colors = np.concatenate([r, g, b, alpha], axis=1).astype(np.uint8)

    x, y, z = colors[:,0], colors[:,1], colors[:,2] 
    pts = quiver3d(x, y, z, ones, ones, ones, scalars=scalars, mode='sphere', scale_factor=1) # Create points
    pts.glyph.color_mode = 'color_by_scalar'

    # Set look-up table and redraw
    pts.module_manager.scalar_lut_manager.lut.table = colors
    mlab.outline()
    draw()
    mlab.show()