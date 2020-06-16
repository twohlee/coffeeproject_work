import matplotlib.pyplot as plt

from skimage import data, color, img_as_ubyte
from skimage.feature import canny
from skimage.transform import hough_ellipse
from skimage.draw import ellipse_perimeter

import cv2
import numpy as np
from PIL import Image
from time import sleep

import glob
import os
import sys

## import coffee green bean image
dataPath = "/home/team/project/GAN/bean/data"
normal_imgs = glob.glob(os.path.join(dataPath+"/normal_rotated_data",'*.jpg'))
broken_imgs = glob.glob(os.path.join(dataPath+"/broken_rotated_data",'*.jpg'))

def test(imgs):
    detected = 0
    Non_detected = 0
    for img in imgs:
        print(detected, Non_detected)
        path = img
        image_rgb = cv2.imread(path, cv2.IMREAD_COLOR)
        image_gray = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        _, image_bin = cv2.threshold(image_gray, 0, 255, cv2.THRESH_OTSU)
        edges = canny(image_bin, sigma=2.0,
                    low_threshold=0.25, high_threshold=0.8)
        try:
            result = hough_ellipse(edges, accuracy=9, threshold=52,
                    min_size=20)

            result.sort(order='accumulator')

            # Estimated parameters for the ellipse
            best = list(result[-1])
            yc, xc, a, b = [int(round(x)) for x in best[1:5]]
            orientation = best[5]

            # Draw the ellipse on the original image
            cy, cx = ellipse_perimeter(yc, xc, a, b, orientation)
            image_rgb[cy, cx] = (0, 0, 255)
            # Draw the edge (white) and the resulting ellipse (red)
            edges = color.gray2rgb(img_as_ubyte(edges))
            edges[cy, cx] = (250, 0, 0)

            detected += 1

        except:
            Non_detected += 1

    return (detected, Non_detected)

norm_det, norm_non = test(normal_imgs)
print(norm_det, norm_non)