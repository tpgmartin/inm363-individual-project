import argparse
from glob import glob
import numpy as np
import os
import pandas as pd
from pathlib import Path
import random
from scipy.spatial import distance
import sys
from tcav import utils
import tcav.model as model
import tensorflow as tf

from helpers import get_acts_from_images, make_model, map_images_to_labels
from concept_discovery import ConceptDiscovery

def main(args):

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

    bn_activations = cd.get_bn_activations(channel_mean='skip')

    print(bn_activations)
    print(len(bn_activations[0]))

    sess.close()

def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--source_dir', type=str,
        help='''Directory where the network's classes image folders and random
        concept folders are saved.''', default='./occluded_images/')
    parser.add_argument('--working_dir', type=str,
        help='Directory to save the results.', default='./')
    parser.add_argument('--model_to_run', type=str,
        help='The name of the model.', default='InceptionV3')
    parser.add_argument('--model_path', type=str,
        help='Path to model checkpoints.', default='./v3_model.h5')
    parser.add_argument('--labels_path', type=str,
        help='Path to model checkpoints.', default='./imagenet_labels.txt')
    parser.add_argument('--target_class', type=str,
        help='The name of the target class to be interpreted', default='jeep')
    parser.add_argument('--bottlenecks', type=str,
        help='Names of the target layers of the network (comma separated)',
                        default='mixed8')
    parser.add_argument('--num_random_exp', type=int,
        help="Number of random experiments used for statistical testing, etc",
                        default=20)
    return parser.parse_args(argv)

if __name__ == '__main__':

    # Get bottleneck encoding for image

    # mapping_images_to_labels = map_images_to_labels()
    args = parse_arguments(sys.argv[1:])
    random_concept = 'random_discovery'
    cavs_dir = os.path.join(args.working_dir, 'cavs/')
    activations_dir = os.path.join(args.working_dir, 'acts/')

    args.source_dir = '../ACE/ImageNet/ILSVRC2012_img_train/n03594945/n03594945_392'

    main(args)
