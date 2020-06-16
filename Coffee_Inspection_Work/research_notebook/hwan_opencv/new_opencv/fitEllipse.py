import matplotlib.pyplot as plt

from skimage import data, color, img_as_ubyte
from skimage.feature import canny
from skimage.transform import hough_ellipse, hough_circle
from skimage.draw import ellipse_perimeter

import cv2
import numpy as np
from PIL import Image
from time import sleep

import glob
import os
import sys

dataPath = "/home/team/project/GAN/bean/data"
normal_imgs = glob.glob(os.path.join(dataPath+"/normal_rotated_data",'*.jpg'))
broken_imgs = glob.glob(os.path.join(dataPath+"/broken_rotated_data",'*.jpg'))

for x in normal_imgs:
    image_rgb = cv2.imread(x, cv2.IMREAD_COLOR)
    image_gray = cv2.imread(x, cv2.IMREAD_GRAYSCALE)
    _, image_bin = cv2.threshold(image_gray, 0, 255, cv2.THRESH_OTSU)

    edges = canny(image_bin, sigma=2.0,
                low_threshold=0.55, high_threshold=0.8)

    test = cv2.fitEllipse(edges)
    cv2.imshow("test", test)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
                
    # try:
    #     result = hough_ellipse(edges, accuracy=9.6, threshold=55,
    #                         min_size=20)
    #     print(result)
    #     # sleep(10)

    #     result.sort(order='accumulator')

    #     # Estimated parameters for the ellipse
    #     best = list(result[-1])
    #     yc, xc, a, b = [int(round(x)) for x in best[1:5]]
    #     orientation = best[5]

    #     # Draw the ellipse on the original image
    #     cy, cx = ellipse_perimeter(yc, xc, a, b, orientation)
    #     image_rgb[cy, cx] = (0, 0, 255)
    #     # Draw the edge (white) and the resulting ellipse (red)
    #     edges = color.gray2rgb(img_as_ubyte(edges))
    #     edges[cy, cx] = (250, 0, 0)


    #     fig = plt.figure(figsize=(20,10))

    #     ax1 = plt.subplot(1,3,1)
    #     ax1.set_title('Original picture')
    #     ax1.imshow(image_rgb)

    #     ax2 = plt.subplot(1,3,2)
    #     ax2.set_title('gray')
    #     ax2.imshow(image_gray)

    #     ax3 = plt.subplot(1,3,3)
    #     ax3.set_title('Edge (white) and result (red)')
    #     ax3.imshow(edges)

    #     plt.show()
    # except:
    #     result = hough_ellipse(edges, accuracy=9.6, threshold=55,
    #                         min_size=20)
    #     print("empty result")
    #     fig = plt.figure(figsize=(20,10))

    #     ax1 = plt.subplot(1,3,1)
    #     ax1.set_title('Original picture')
    #     ax1.imshow(image_rgb)

    #     ax2 = plt.subplot(1,3,2)
    #     ax2.set_title('gray')
    #     ax2.imshow(image_gray)

    #     ax3 = plt.subplot(1,3,3)
    #     ax3.set_title('Edge (white) and result (red)')
    #     ax3.imshow(edges)

    #     plt.show()