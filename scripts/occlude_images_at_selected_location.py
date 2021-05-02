from glob import glob
import itertools
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import random
from skimage import draw
from skimage import io
import time

from helpers import map_labels_to_dirs

def main(img_filename, dir_name, x, y, MAX_MASK_SIZE):

    IMG = io.imread(img_filename)
    HEIGHT, WIDTH, _ = IMG.shape
    DPI = 72 # Check this value
    MASK_SIZE = np.min([(np.min([HEIGHT, WIDTH]) // 100)*100, MAX_MASK_SIZE])

    # Mask colour
    mean_r_channel = np.mean([item[0]/255 for sublist in IMG for item in sublist])
    mean_g_channel = np.mean([item[1]/255 for sublist in IMG for item in sublist])
    mean_b_channel = np.mean([item[2]/255 for sublist in IMG for item in sublist])
    mask_colour = [mean_r_channel, mean_g_channel, mean_b_channel]

    coords = (y, x)
    img_filename_to_save = img_filename.split('/')[-1].split('.')[0]

    fig, ax = plt.subplots(figsize=((IMG.shape[1]/DPI),(IMG.shape[0]/DPI)), dpi=DPI)
    ax.imshow(IMG)    

    mask = draw.rectangle((coords[0],coords[1]), extent=(MASK_SIZE,MASK_SIZE))
    ax.plot(mask[1], mask[0], color=mask_colour, lw=1)

    ax.axis('off')
    plt.tight_layout(pad=0)
    plt.savefig(f'{dir_name}/{img_filename_to_save}_occluded_1.JPEG')
    plt.clf()


if __name__ == '__main__':

    MAX_MASK_SIZE = 100

    imgs_occlusion_coords = {
        'n03594945_571': {
            'x': 60,
            'y': 250
        },
        'n03594945_3488': {
            'x': 315,
            'y': 110
        },
        'n03594945_9292': {
            'x': 520,
            'y': 460
        },
        'n03594945_392': {
            'x': 60,
            'y': 300
        },
        'n03594945_289': {
            'x': 325,
            'y': 215
        }
    }

    for image in glob('../ACE/ImageNet/jeep_occlusion_images/*'):
        try:
            img_code = image.split('/')[-1].split('.')[0]
            start = time.time()
            print(f'Starting image: {image}')
            main(image, '/'.join(image.split('/')[:-1]), imgs_occlusion_coords[img_code]['x'], imgs_occlusion_coords[img_code]['y'], MAX_MASK_SIZE)
            print(f'Finsihed image: {image}')
            end = time.time()
            print('Elapsed time (minutes):', (end - start)/60)
        except KeyError:
            pass
