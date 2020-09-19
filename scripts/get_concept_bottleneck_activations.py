import argparse
from glob import glob
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from pathlib import Path
from PIL import Image
from skimage import io
import sklearn.cluster as cluster
import sys
from tcav import utils
import tcav.model as model
import tensorflow as tf

from helpers import make_model
from concept_discovery import ConceptDiscovery

def get_patch_activations(args):
    
    sess = utils.create_session()
    mymodel = make_model(sess, args.model_to_run, args.model_path, args.labels_path)

    cd = ConceptDiscovery(
        mymodel,
        None,
        args.bottlenecks,
        sess,
        args.source_dir)

    bn_activations = cd.get_bn_activations()
    
    sess.close()

    return bn_activations

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
    try:
        x_min, x_max = np.min(x_coords), np.max(x_coords)
        y_min, y_max = np.min(y_coords), np.max(y_coords)
    except ValueError:
        # No image segments that boost prediction probability - so exit here
        return

    # Crop input image to dimensions of image segment
    cropped_image = []
    for y in range(y_min,y_max):
        cropped_image.append([])
        for x in range(x_min,x_max):
            cropped_image[y-y_min].append([])
            cropped_image[y-y_min][x-x_min].append(IMG[y][x][0])
            cropped_image[y-y_min][x-x_min].append(IMG[y][x][1])
            cropped_image[y-y_min][x-x_min].append(IMG[y][x][2])

    # Change name "patch"
    filepath = f"./patches/{IMAGE_CAT}/{IMAGE_NO}/mask_dim_{MASK_DIM}/{IMAGE_NO}_superpixels"
    os.makedirs(filepath, exist_ok=True)

    # Save patch resized to original image dimensions
    cropped_image = Image.fromarray((np.array(cropped_image)).astype(np.uint8))
    image_resized = np.array(cropped_image.resize([IMG.shape[1], IMG.shape[0]], Image.BICUBIC))
    # Change name "patch"
    Image.fromarray(image_resized).save(f"{filepath}/{IMAGE_NO}_superpixels.png", format='PNG')

    return image_resized, cropped_image, filepath

def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--source_dir', type=str,
        help='''Directory where the network's classes image folders and random
        concept folders are saved.''', default='')
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
    # TODO: Need to indentify all seperate segments of given image e.g. in cases where image segments are not contiguous

    # 3. Find activation for each patch - this is pretty much what I've already done, but would be applied to patches from previous steps only 
    # 4. Apply clustering - just standard k-means
    # 5. Remove outliers from clustering technique - this creates bottleneck dictionary for given bottleneck layer

    # 6. Discover CAVs
    filepaths = []
    images = []
    patches = []
    bn_activations = []
    # labels = []
    # costs = []

    # "bubble" images only
    images = glob('./net_occlusion_heatmaps_delta_prob/n09229709/**/**/*_image_cropped_to_mask/*')[:3]
    for image in images:
    # img_path = './net_occlusion_heatmaps_delta_prob/n09229709/n09229709_47343/mask_dim_100/n09229709_47343_image_cropped_to_mask/n09229709_47343_image_cropped_to_mask.JPEG'
        print('------------------------------')
        print(image)
        print('------------------------------')
        superpixel, patch, filepath = extract_patch(image)
        if filepath:
            filepaths.append(filepath)
            args = parse_arguments(sys.argv[1:])
            args.source_dir = filepath
            # Get activation for superpixel
            bn_activations.append(get_patch_activations(args)[0])

    # Save to CSV

    # Cluster activations
    # * Perform clustering on bottleneck activations using given method
    # * cluster method: use k-means
    # * (Check format of bn_dic['label'])
    # * For given label in bn_activations, if have sufficient number of images (at least 20)
    # * Check if concept images meet 1 of 3 conditons:
    #     - Concept found in more than half of images of given label
    #     - Concept found in more than quarter of images of given label AND number of supporting 
    # images more quarter of all images
    #     - Concept found in more than 10% of images of given label AND number of supporting 
    # images more half of all images
    # * If true then for each concept reutrn images, partcher, and image numbers

    centers = None
    n_clusters = 3
    km = cluster.KMeans(n_clusters)
    d = km.fit(bn_activations)
    centers = km.cluster_centers_
    d = np.linalg.norm(np.expand_dims(bn_activations, 1) - np.expand_dims(centers, 0), ord=2, axis=-1)
    labels, costs = np.argmin(d, -1), np.min(d, -1)

    # label concepts by target label and concept number
    # if highly_common_concept or cond2 or cond3:
    #         concept_number += 1
    #         concept = '{}_concept{}'.format(self.target_class, concept_number)
    #         bn_dic['concepts'].append(concept)
    #         bn_dic[concept] = {
    #             'images': self.dataset[concept_idxs],
    #             'patches': self.patches[concept_idxs],
    #             'image_numbers': self.image_numbers[concept_idxs]
    #         }
    #         bn_dic[concept + '_center'] = centers[i]

    concepts = []
    for l in range(len(labels)):
        concept = {}
        concept['concept'] = l
        concept['center'] = centers[l]
        idxs = [idx for idx, x in enumerate(labels) if x == l]
        concept['images'] = [filepaths[idx] for idx in idxs]
        concepts.append(concept)

    print('+++++++++++++++++++++++++++++++++++++')
    print(concepts)
    print('+++++++++++++++++++++++++++++++++++++')
    # ^ Save labels and centers


