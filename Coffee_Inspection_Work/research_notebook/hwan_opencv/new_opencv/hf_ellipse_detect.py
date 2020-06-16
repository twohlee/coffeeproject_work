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

# Load picture, convert to grayscale and detect edges
# image_rgb = data.coffee()[0:220, 160:420]
# print(type(image_rgb)) # <class 'numpy.ndarray'>
# print(image_rgb.shape) # (220, 260, 3)

## import coffee green bean image
dataPath = "/home/team/project/GAN/bean/data"
normal_imgs = glob.glob(os.path.join(dataPath+"/normal_rotated_data",'*.jpg'))
broken_imgs = glob.glob(os.path.join(dataPath+"/broken_rotated_data",'*.jpg'))

# x_list = range(1,13)
# for x in x_list:
for x in normal_imgs:
    # path = './data/normal/'+ str(x) +'.jpg'
    # path = './data/broken/'+ str(x) +'.jpg'
    image_rgb = cv2.imread(x, cv2.IMREAD_COLOR)
    image_gray = cv2.imread(x, cv2.IMREAD_GRAYSCALE)
    _, image_bin = cv2.threshold(image_gray, 0, 255, cv2.THRESH_OTSU)


    # cv2.imshow("image_rgb",image_rgb)
    # cv2.imshow("image_gray",image_gray)
    # cv2.imshow("image_bin",image_bin)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    edges = canny(image_bin, sigma=2.0,
                low_threshold=0.55, high_threshold=0.8)
    # for i in edges:
    #     for j in i:
    #         if j:
    #             print(j)
    # print("out")
    # sleep(10)
    # print(edges)
    # sleep(10)

    # fig = plt.figure(figsize=(20,10))

    # ax1 = plt.subplot(1,3,1)
    # ax1.set_title('Original picture')
    # ax1.imshow(image_rgb)

    # ax2 = plt.subplot(1,3,2)
    # ax2.set_title('gray')
    # ax2.imshow(image_gray)

    # ax3 = plt.subplot(1,3,3)
    # ax3.set_title('Edge (white) and result (red)')
    # ax3.imshow(edges)

    # plt.show()
    # sleep(1)


    # Perform a Hough Transform
    # The accuracy corresponds to the bin size of a major axis.
    # The value is chosen in order to get a single high accumulator.
    # The threshold eliminates low accumulators
    try:
        result = hough_ellipse(edges, accuracy=9.6, threshold=55,
                            min_size=20)
        print(result)
        # sleep(10)

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

        # fig2, (ax1, ax2) = plt.subplots(ncols=2, nrows=1, figsize=(8, 4),
        #                                 sharex=True, sharey=True)

        fig = plt.figure(figsize=(20,10))

        ax1 = plt.subplot(1,3,1)
        ax1.set_title('Original picture')
        ax1.imshow(image_rgb)

        ax2 = plt.subplot(1,3,2)
        ax2.set_title('gray')
        ax2.imshow(image_gray)

        ax3 = plt.subplot(1,3,3)
        ax3.set_title('Edge (white) and result (red)')
        ax3.imshow(edges)

        plt.show()
    except:
        result = hough_ellipse(edges, accuracy=9.6, threshold=55,
                            min_size=20)
        print("empty result")
        fig = plt.figure(figsize=(20,10))

        ax1 = plt.subplot(1,3,1)
        ax1.set_title('Original picture')
        ax1.imshow(image_rgb)

        ax2 = plt.subplot(1,3,2)
        ax2.set_title('gray')
        ax2.imshow(image_gray)

        ax3 = plt.subplot(1,3,3)
        ax3.set_title('Edge (white) and result (red)')
        ax3.imshow(edges)

        plt.show()