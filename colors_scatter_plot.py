# Built-in imports
import random
import sys
import argparse

# External imports
import cv2
import numpy as np
import vispy.scene
from vispy.color import ColorArray


def scale(arr):
    return (arr - np.min(arr)) / (np.max(arr) - np.min(arr))


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
        x,y,z = r/255, g/255, b/255
        color1, color2, color3 = x, y, z
    
    elif args['colormap'] == 'hsv':
        # OpenCV uses HSV ranges between (0-180, 0-255, 0-255)
        im = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
        
        # In HSV, H is the theta, S is the radius and V is the altitude
        h,s,v = cv2.split(im)
        # theta, radius, altitude = h,s,v
        #x = radius * np.cos(np.deg2rad(2 * theta))
        #y = radius * np.sin(np.deg2rad(2 * theta))
        #z = altitude
        
        radius = np.random.rand(N, 1)
        theta = np.deg2rad(360 * np.random.rand(N, 1))
        altitude = np.random.rand(N, 1)
        
        x = radius * np.cos(theta)
        y = radius * np.sin(theta)
        z = altitude

        color1 = scale(theta)
        color2 = radius
        color3 = altitude

    else:
        print(f"Only valid colormaps are 'rgb' and 'hsv'")
        sys.exit()

    x = x.reshape((N, 1))
    y = y.reshape((N, 1))
    z = z.reshape((N, 1))

    color1 = color1.reshape((N, 1))
    color2 = color2.reshape((N, 1))
    color3 = color3.reshape((N, 1))

    # Vispy
    Scatter3D = vispy.scene.visuals.create_visual_node(vispy.visuals.MarkersVisual)

    canvas = vispy.scene.SceneCanvas(keys='interactive', show=True)
    view = canvas.central_widget.add_view()
    # SHIFT + LMB to translate the center point
    view.camera = 'turntable' # 'base', 'arcball', 'turntable', 'panzoom', 'fly'
    view.camera.distance = 5

    pos = np.concatenate([x,y,z], axis=1)
    colors = ColorArray(color=np.concatenate([color1, color2, color3], axis=1), color_space=args['colormap'])
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