import argparse
from glob import glob
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from pathlib import Path
from PIL import Image
from skimage import io
import sys
from tcav import utils
import tcav.model as model
import tensorflow as tf

from helpers import make_model
from concept_discovery import ConceptDiscovery

def main(args):
    
    sess = utils.create_session()
    mymodel = make_model(sess, args.model_to_run, args.model_path, args.labels_path)

    cd = ConceptDiscovery(
        mymodel,
        None,
        args.bottlenecks,
        sess,
        args.source_dir)

    # try:
    #     prediction, filename = cd.predict()
    #     predictions.append(prediction)
    #     filenames.append(filename[0])
    # except ValueError as e:
    #     predictions.append(np.nan)
    #     filenames.append(source_dir)
    #     pass
    
    sess.close()


def parse_arguments(argv):
  parser = argparse.ArgumentParser()
  parser.add_argument('--source_dir', type=str,
      help='''Directory where the network's classes image folders and random
      concept folders are saved.''', default='./net_occlusion_heatmaps_delta_prob/n02966193/n02966193_2173/mask_dim_100/n02966193_2173_image_cropped_to_mask/')
  parser.add_argument('--model_to_run', type=str,
      help='The name of the model.', default='GoogleNet')
  parser.add_argument('--model_path', type=str,
      help='Path to model checkpoints.', default='./tensorflow_inception_graph.pb')
  parser.add_argument('--labels_path', type=str,
      help='Path to model checkpoints.', default='./imagenet_labels.txt')
  parser.add_argument('--bottlenecks', type=str,
      help='Names of the target layers of the network (comma separated)',
                      default='mixed4c')
  return parser.parse_args(argv)


def extract_patch(image, DPI=72):

    IMG = io.imread(image)

    HEIGHT, WIDTH, _ = IMG.shape
    DPI = DPI
    IMAGE_NO = image.split('/')[3]
    IMAGE_CAT = IMAGE_NO.split('_')[0]
    MASK_DIM = image.split('/')[4].split('_')[-1]

    x_coords = []
    y_coords = []
    for y in range(HEIGHT):
        for x in range(WIDTH):
            if np.mean(IMG[y][x])>0:
                x_coords.append(x)
                y_coords.append(y)

    # Get x,y-coords of image segment
    x_min, x_max = np.min(x_coords), np.max(x_coords)
    y_min, y_max = np.min(y_coords), np.max(y_coords)

    # Crop input image to dimensions of image segment
    cropped_image = []
    for y in range(y_min,y_max):
        cropped_image.append([])
        for x in range(x_min,x_max):
            cropped_image[y-y_min].append([])
            cropped_image[y-y_min][x-x_min].append(IMG[y][x][0])
            cropped_image[y-y_min][x-x_min].append(IMG[y][x][1])
            cropped_image[y-y_min][x-x_min].append(IMG[y][x][2])

    os.makedirs(f"./patches/{IMAGE_CAT}/{IMAGE_NO}/mask_dim_{MASK_DIM}/{IMAGE_NO}_patch", exist_ok=True)

    # Save patch resized to original image dimensions
    cropped_image = Image.fromarray((np.array(cropped_image)).astype(np.uint8))
    image_resized = np.array(cropped_image.resize([IMG.shape[1], IMG.shape[0]], Image.BICUBIC))
    Image.fromarray(image_resized).save(f"./patches/{IMAGE_CAT}/{IMAGE_NO}/mask_dim_{MASK_DIM}/{IMAGE_NO}_patch/{IMAGE_NO}_patch.png", format='PNG')

if __name__ == '__main__':

    # 1. Need to find patches from heatmap - likely just elements that are not blacked out
    # 2. Resize each patch to original input image size

    # ^ For these steps just want to call somthing like `_extract_patch`

    # 3. Find activation for each patch - this is pretty much what I've already done, but would be applied to patches from previous steps only 
    # 4. Apply clustering - just standard k-means
    # 5. Remove outliers from clustering technique - this creates bottleneck dictionary for given bottleneck layer

    # main(parse_arguments(sys.argv[1:]))

    # Resize image patch

    img_path = './net_occlusion_heatmaps_delta_prob/n09229709/n09229709_47343/mask_dim_100/n09229709_47343_image_cropped_to_mask/n09229709_47343_image_cropped_to_mask.JPEG'
    extract_patch(img_path)