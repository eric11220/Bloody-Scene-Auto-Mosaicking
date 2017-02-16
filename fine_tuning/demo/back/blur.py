#-*- coding: utf-8 -*-
import os
import sys
import time
import cv2
import scipy
import math
import random
caffe_root = '../../../caffe/'
sys.path.insert(0, caffe_root + 'python')
import caffe
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageFilter, ImageChops
BLUE = 0
GREEN = 1
RED = 2

class GaussianBlur(ImageFilter.Filter):
    name = "GaussianBlur"

    def __init__(self, radius=2, bounds=None):
        self.radius = radius
        self.bounds = bounds

    def filter(self, image):
        if self.bounds:
            clips = image.crop(self.bounds).gaussian_blur(self.radius)
            image.paste(clips, self.bounds)
            return image
        else:
            return image.gaussian_blur(self.radius)

def blur(img, mask):
    w = img.size[0]
    h = img.size[1]
    mask = cv2.resize(mask, (w, h))

    blue_channel = mask[:, :, BLUE]
    green_channel = mask[:, :, GREEN]
    # Get region to be blurred
    blur_region = np.zeros((h, w))
    blur_region = np.bitwise_and(blue_channel == 0, green_channel < 200)
    blur_region = blur_region[:, :, np.newaxis]
    blur_region = np.tile(blur_region, (1, 1, 3))

    # Blur all image
    b = img.filter(GaussianBlur(radius=30, bounds=(0, 0, w, h)))

    # Get background and blur foreground
    background = np.multiply(-blur_region, img)
    background = Image.fromarray(background)
    foreground = np.multiply(blur_region, b)
    foreground = Image.fromarray(foreground)
    # Merge
    img = ImageChops.add(foreground, background, scale=1.0, offset=0)

    return img

def main(argv):
    start = time.time()

    #mask = cv2.imread(argv[2])

    # Blur image with given mask
    img = Image.open(argv[1])
    img = img.filter(GaussianBlur(radius=30, bounds=(0, 0, img.size[0], img.size[1])))
    #img = blur(img, mask)

    # Output image
    img.save('_output.png')
    end = time.time()

    print str(end - start) + ' sec'

# Start point if this script is main program
if __name__ == '__main__':
    main(sys.argv)
