#-*- coding: utf-8 -*-
import os
import sys
import time
import cv2
import scipy
caffe_root = '../../caffe/'
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

    # Get background and blur foreground
    background = np.multiply(-blur_region, img)
    background = Image.fromarray(background)
    foreground = np.multiply(blur_region, img)
    foreground = Image.fromarray(foreground)
    foreground = foreground.filter(GaussianBlur(radius=29, bounds=(0, 0, w, h)))

    # Merge foreground, background
    img = ImageChops.add(foreground, background, scale=1.0, offset=0)
    return img

def main(argv):
    start = time.time()

    net_full_conv = caffe.Net('/tmp3/changjenyin/caffe/examples/imagenet/bvlc_caffenet_full_conv.prototxt', '/tmp3/changjenyin/caffe/examples/imagenet/bvlc_caffenet_full_conv.caffemodel')
    net_full_conv.set_phase_test()
    net_full_conv.set_mean('data', np.load('/tmp3/changjenyin/fine_tuning/mean.npy'))
    net_full_conv.set_channel_swap('data', (2,1,0))
    net_full_conv.set_raw_scale('data', 255.0)

    read_fifo = open('main_to_script.fifo', 'r')
    write_fifo = open('script_to_main.fifo', 'w')
    while 1:
        img_path = read_fifo.readline().replace('\n', '')
        if os.path.isfile(img_path) == False:
            write_fifo.write("File not exists!\n")
            write_fifo.flush()
            continue
        img_name = os.path.basename(img_path)
        img_name = os.path.splitext(img_name)[0]
        print img_path
        print img_name
        write_fifo.write("got it\n")
        write_fifo.flush()


        # Load net, image
        img = caffe.io.load_image(img_path)

        # Make classification map by forward and print prediction indices at each location
        out = net_full_conv.forward_all(data=np.asarray([net_full_conv.preprocess('data', img)]))

        # Extract classification map as mask
        fig = plt.figure(figsize=(16, 9), frameon=False, dpi=80)
        ax = plt.Axes(fig, [0, 0, 1, 1])
        ax.set_axis_off()
        fig.add_axes(ax)

        output_dir = 'masque'
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)

        mask = output_dir+'/'+img_name+'_mask.png'

        plt.imshow(out['prob'][0][1], aspect='normal')
        plt.savefig(mask, dpi=80)

        mask = cv2.imread(mask)

        # Blur image with given mask
        img = Image.open(img_path)
        img = blur(img, mask)

        # Output image
        image = output_dir+'/'+img_name+'_output.png'
        img.save(image)
        end = time.time()

        write_fifo.write("done\n")
        write_fifo.flush()
        print str(end - start) + ' sec'

# Start point if this script is main program
if __name__ == '__main__':
    main(sys.argv)
