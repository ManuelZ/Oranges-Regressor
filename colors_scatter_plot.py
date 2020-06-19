# Built-in imports
import random

# External imports
import numpy as np
import cv2
import vispy.scene
from vispy.color import ColorArray

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

    # Vispy
    Scatter3D = vispy.scene.visuals.create_visual_node(vispy.visuals.MarkersVisual)

    canvas = vispy.scene.SceneCanvas(keys='interactive', show=True)
    view = canvas.central_widget.add_view()
    # SHIFT + LMB to translate the center point
    view.camera = 'arcball'  # 'base', 'arcball', 'turntable', 'panzoom', 'fly'
    view.camera.distance = 2

    pos = np.concatenate([r,g,b], axis=1) / 255
    colors = np.concatenate([r,g,b], axis=1) / 255
    sizes = 5 * np.ones((N,1))

    p1 = Scatter3D(parent=view.scene)
    p1.set_gl_state('translucent', blend=True, depth_test=True)
    p1.set_data(pos,
                face_color=colors,
                symbol='o',
                size=sizes[:,0],
                edge_width=0.5, 
                edge_color=None
    )

    axis = vispy.scene.visuals.XYZAxis(parent=view.scene)
    vispy.app.run()