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
    IMG = io.imread(img_path)

    HEIGHT, WIDTH, _ = IMG.shape
    DPI = 72

    x_coords = []
    y_coords = []
    for y in range(HEIGHT):
        for x in range(WIDTH):
            if np.mean(IMG[y][x])>0:
                x_coords.append(x)
                y_coords.append(y)

    # h1, h2, w1, w2 = ones[0].min(), ones[0].max(), ones[1].min(), ones[1].max()
    x_min, x_max = np.min(x_coords), np.max(x_coords)
    y_min, y_max = np.min(y_coords), np.max(y_coords)

    # IMG[y][x] = [c1,c2,c3]

    cropped_image = []
    for y in range(y_min,y_max):
        cropped_image.append([])
        for x in range(x_min,x_max):
            cropped_image[y-y_min].append([])
            cropped_image[y-y_min][x-x_min].append(IMG[y][x][0])
            cropped_image[y-y_min][x-x_min].append(IMG[y][x][1])
            cropped_image[y-y_min][x-x_min].append(IMG[y][x][2])

    fig, ax = plt.subplots(figsize=(((x_max-x_min)/DPI),((y_max-y_min)/DPI)), dpi=DPI)
    ax.imshow(cropped_image)
    ax.axis('off')
    plt.tight_layout(pad=0)

    plt.savefig("cropped_example.JPEG")

    # image_resized = np.array(image.resize(self.image_shape, Image.BICUBIC)).astype(float) / 255
    cropped_image = Image.fromarray((np.array(cropped_image)).astype(np.uint8))
    image_resized = np.array(cropped_image.resize(IMG.shape[:2], Image.BICUBIC)).astype(float) / 255

    Image.fromarray(image_resized).save('resized_example', format='PNG')

    # fig, ax = plt.subplots(figsize=((WIDTH/DPI),(HEIGHT/DPI)), dpi=DPI)
    # ax.imshow(cropped_image)
    # ax.axis('off')
    # plt.tight_layout(pad=0)

    # plt.savefig("resized_example.JPEG")

    # def _extract_patch(self, image, mask):
    """Extracts a patch out of an image.

    Args:
      image: The original image
      mask: The binary mask of the patch area

    Returns:
      image_resized: The resized patch such that its boundaries touches the
        image boundaries
      patch: The original patch. Rest of the image is padded with average value
    """
    # Find x, y coords of non-zero pixels
    # Crop image to these coords
    # Resize cropped iamge to original image size
    # mask_expanded = np.expand_dims(mask, -1)
    # patch = (mask_expanded * image + (
    #     1 - mask_expanded) * float(self.average_image_value) / 255)
    # ones = np.where(mask == 1)
    # h1, h2, w1, w2 = ones[0].min(), ones[0].max(), ones[1].min(), ones[1].max()
    # image = Image.fromarray((patch[h1:h2, w1:w2] * 255).astype(np.uint8))
    # image_resized = np.array(image.resize(self.image_shape,
    #                                       Image.BICUBIC)).astype(float) / 255