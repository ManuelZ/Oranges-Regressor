# Built-in imports
import random
import sys
import argparse

# External imports
import cv2
import numpy as np
import vispy.scene
from vispy.color import ColorArray


if __name__ == '__main__':
    
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", required=True, type=str, help="Filename of input image")
    ap.add_argument("-c", "--colormap", required=True, type=str, help="Either rgb or hsv")
    args = vars(ap.parse_args())

    im = cv2.imread(args['image'])
    
    # Resize image
    target_width = 300
    ratio = target_width / im.shape[1]
    dim = (int(target_width), int(im.shape[0] * ratio))
    im = cv2.resize(im, dim, interpolation = cv2.INTER_LINEAR_EXACT)
    
    N = im.shape[0] * im.shape[1] # Number of points

    if args['colormap'] == 'rgb':
        b,g,r = cv2.split(im)
        x,y,z = r,g,b
    elif args['colormap'] == 'hsv':
        im = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
        h,s,v = cv2.split(im)
        x,y,z = h,s,v
    else:
        print(f'Only valid colormaps are RGB and HSV')
        sys.exit()


    x = x.reshape((N, 1)).astype(np.uint8)
    y = y.reshape((N, 1)).astype(np.uint8)
    z = z.reshape((N, 1)).astype(np.uint8)
    
    # Vispy
    Scatter3D = vispy.scene.visuals.create_visual_node(vispy.visuals.MarkersVisual)

    canvas = vispy.scene.SceneCanvas(keys='interactive', show=True)
    view = canvas.central_widget.add_view()
    # SHIFT + LMB to translate the center point
    view.camera = 'turntable' # 'base', 'arcball', 'turntable', 'panzoom', 'fly'
    view.camera.distance = 2

    pos = np.concatenate([x,y,z], axis=1) / 255
    colors = ColorArray(color_space=args['colormap'], color=(np.concatenate([x,y,z], axis=1) / 255))
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