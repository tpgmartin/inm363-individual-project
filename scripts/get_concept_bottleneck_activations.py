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

def get_patch_activations(args, activations_dir, cavs_dir, random_concept='random_discovery'):
    
    sess = utils.create_session()
    mymodel = make_model(sess, args.model_to_run, args.model_path, args.labels_path)

    cd = ConceptDiscovery(
        mymodel,
        args.target_class,
        random_concept,
        args.bottlenecks,
        sess,
        args.source_dir,
        activations_dir,
        cavs_dir)

    bn_activations = cd.get_bn_activations()
    
    sess.close()

    return bn_activations

def extract_patch(image, DPI=72):

    IMG = io.imread(image)

    HEIGHT, WIDTH, _ = IMG.shape
    DPI = DPI # <- skip this line
    IMAGE_NO = image.split('/')[3] # <- skip this line
    IMAGE_CAT = IMAGE_NO.split('_')[0] # <- skip this line
    MASK_DIM = image.split('/')[4].split('_')[-1] # <- skip this line

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

    filepath = f"./superpixels/{IMAGE_CAT}/{IMAGE_NO}/mask_dim_{MASK_DIM}/{IMAGE_NO}_superpixels" # <- skip this line
    os.makedirs(filepath, exist_ok=True) # <- skip this line

    # Save patch resized to original image dimensions
    cropped_image = Image.fromarray((np.array(cropped_image)).astype(np.uint8))
    image_resized = np.array(cropped_image.resize([IMG.shape[1], IMG.shape[0]], Image.BICUBIC))
    Image.fromarray(image_resized).save(f"{filepath}/{IMAGE_NO}_superpixels.png", format='PNG') # <- skip this line
    # "cropped_image" is same as "patch"
    return image_resized, cropped_image, filepath

def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--source_dir', type=str,
        help='''Directory where the network's classes image folders and random
        concept folders are saved.''', default='')
    parser.add_argument('--working_dir', type=str,
      help='Directory to save the results.', default='./')
    parser.add_argument('--model_to_run', type=str,
        help='The name of the model.', default='GoogleNet')
    parser.add_argument('--model_path', type=str,
        help='Path to model checkpoints.', default='./tensorflow_inception_graph.pb')
    parser.add_argument('--labels_path', type=str,
        help='Path to model checkpoints.', default='./imagenet_labels.txt')
    parser.add_argument('--target_class', type=str,
      help='The name of the target class to be interpreted', default='bubble')
    parser.add_argument('--bottlenecks', type=str,
        help='Names of the target layers of the network (comma separated)',
                        default='mixed4c')
    parser.add_argument('--num_random_exp', type=int,
        help="Number of random experiments used for statistical testing, etc",
                        default=20)
    return parser.parse_args(argv)

if __name__ == '__main__':

    args = parse_arguments(sys.argv[1:])
    random_concept = 'random_discovery'
    cavs_dir = os.path.join(args.working_dir, 'cavs/')
    activations_dir = os.path.join(args.working_dir, 'acts/')
    tf.gfile.MakeDirs(cavs_dir)
    tf.gfile.MakeDirs(activations_dir)

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
    images = glob('./net_occlusion_heatmaps_delta_prob/n09229709/**/**/*_image_cropped_to_mask/*')
    for image in images:
    # img_path = './net_occlusion_heatmaps_delta_prob/n09229709/n09229709_47343/mask_dim_100/n09229709_47343_image_cropped_to_mask/n09229709_47343_image_cropped_to_mask.JPEG'
        
        extract_patch(image)
        
        # superpixel, patch, filepath = extract_patch(image)
        # if filepath:
        #     filepaths.append(filepath)
        #     args.source_dir = filepath
        #     # Get activation for superpixel
        #     bn_activations.append(get_patch_activations(args, activations_dir, cavs_dir)[0])

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

    # What can I do to reuse original ACE code?

    # Skip for now
    # centers = None
    # n_clusters = 3 # Need to adjust this
    # km = cluster.KMeans(n_clusters)
    # d = km.fit(bn_activations)
    # centers = km.cluster_centers_
    # d = np.linalg.norm(np.expand_dims(bn_activations, 1) - np.expand_dims(centers, 0), ord=2, axis=-1)
    # labels, costs = np.argmin(d, -1), np.min(d, -1)

    # concepts = []
    # for c in range(len(centers)):
    #     concept = {}
    #     concept['concept'] = f"{args.target_class}_concept{c}"
    #     concept['center'] = centers[c]
    #     idxs = [idx for idx, x in enumerate(labels) if x == c]
    #     concept['images'] = [filepaths[idx] for idx in idxs]
    #     concepts.append(concept)

    # sess = utils.create_session()
    # mymodel = make_model(sess, args.model_to_run, args.model_path, args.labels_path)

    # for concept in concepts:

    #     concept_acts = []
    #     for concept_img in concept['images']:

    #         cd = ConceptDiscovery(
    #             mymodel,
    #             args.target_class,
    #             random_concept,
    #             args.bottlenecks,
    #             sess,
    #             f"{concept_img}/",
    #             activations_dir,
    #             cavs_dir,
    #             num_random_exp=args.num_random_exp)

    #         cav_accuraciess, concepts_to_delete = cd.cavs(concept)

    #     # get TCAVs
    #     print('tcavs ~~~~~~~~~~~~~~~~~~~~~~~~~')
    #     print(cd.tcavs(concept))
    #     print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
    
    # sess.close()
