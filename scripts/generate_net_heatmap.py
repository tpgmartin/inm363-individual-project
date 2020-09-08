from glob import glob
import matplotlib.pyplot as plt
import os
import pandas as pd
import random
import re
from skimage import io

from helpers import map_images_to_labels

def init_mask(width, height):

    img_mask = []
    for y in range(height):
        img_mask.append([])
        for _ in range(width):
            img_mask[y].append([0,0,0])
    
    return img_mask

def save_image(image_to_save, image_filename_to_save, width, height, dpi, img_filename, MASK_SIZE):

    fig, ax = plt.subplots(figsize=((width/dpi),(height/dpi)), dpi=dpi)
    ax.imshow(image_to_save)
    ax.axis('off')
    plt.tight_layout(pad=0)

    os.makedirs(f"./net_occlusion_heatmaps/{img_filename.split('_')[0]}/{img_filename}/mask_dim_{MASK_SIZE}/{img_filename}_{image_filename_to_save}", exist_ok=True)
    plt.savefig(f"./net_occlusion_heatmaps/{img_filename.split('_')[0]}/{img_filename}/mask_dim_{MASK_SIZE}/{img_filename}_{image_filename_to_save}/{img_filename}_{image_filename_to_save}.JPEG")

def main(f, label):


    try:
        false_heatmap = io.imread(f"./occlusion_heatmaps/{f.split('_')[0]}/{f}/mask_dim_100/is_prediction_correct_false/{f}_mask_heatmap/{f}_mask_heatmap.JPEG")
        true_heatmap = io.imread(f"./occlusion_heatmaps/{f.split('_')[0]}/{f}/mask_dim_100/is_prediction_correct_true/{f}_mask_heatmap/{f}_mask_heatmap.JPEG")
        IMG = io.imread(f"../ACE/ImageNet/ILSVRC2012_img_train/{f.split('_')[0]}/{f}/{f}.JPEG")
    except FileNotFoundError:
        return

    HEIGHT, WIDTH, _ = IMG.shape
    DPI = 72
    MASK_SIZE = 100

    img_mask = init_mask(WIDTH, HEIGHT)

    max_pos_intensity = 0
    min_pos_intensity = 0
    max_neg_intensity = 0
    min_neg_intensity = 0
    for y in range(HEIGHT):
        for x in range(WIDTH):
            net_intensity = int(false_heatmap[y][x][0]) - int(true_heatmap[y][x][0])

            img_mask[y][x][0] = net_intensity

            if net_intensity >= 0:

                if net_intensity > max_pos_intensity:
                    max_pos_intensity = net_intensity

                if net_intensity < min_pos_intensity:
                    min_pos_intensity = net_intensity
            
            else:

                if abs(net_intensity) > max_neg_intensity:
                    max_neg_intensity = abs(net_intensity)

                if abs(net_intensity) < min_neg_intensity:
                    min_neg_intensity = abs(net_intensity)
    
    for y in range(HEIGHT):
        for x in range(WIDTH):
            intensity = img_mask[y][x][0]

            if intensity >= 0:
                normalised_intensity = (intensity - min_pos_intensity)/(max_pos_intensity - min_pos_intensity)
                img_mask[y][x][0] = normalised_intensity
            else:
                normalised_intensity = (abs(intensity) - min_neg_intensity)/(max_neg_intensity - min_neg_intensity)
                img_mask[y][x][0] = 0
                img_mask[y][x][2] = normalised_intensity

            IMG[y][x][0] = (1-img_mask[y][x][0]-img_mask[y][x][2])*IMG[y][x][0] + 255*img_mask[y][x][0]
            IMG[y][x][1] = (1-img_mask[y][x][0]-img_mask[y][x][2])*IMG[y][x][1]
            IMG[y][x][2] = (1-img_mask[y][x][0]-img_mask[y][x][2])*IMG[y][x][2] + 255*img_mask[y][x][2]

    save_image(img_mask, 'net_heatmap', WIDTH, HEIGHT, DPI, f, MASK_SIZE)
    save_image(IMG, 'image_with_mask', WIDTH, HEIGHT, DPI, f, MASK_SIZE)
    plt.clf()

if __name__ == '__main__':

    mapping = map_images_to_labels()

    occlusion_heatmaps = [f.split('/')[-1] for f in glob('./occlusion_heatmaps/**/*')]
    occlusion_heatmaps = [f for f in occlusion_heatmaps if '_' in f]
    existing_heatmaps = [f.split('/')[-1] for f in glob('./net_occlusion_heatmaps/**/*')]
    heatmaps = list(set(occlusion_heatmaps) - set(existing_heatmaps))

    for f in heatmaps:
        main(f, mapping[f.split('_')[0]])
