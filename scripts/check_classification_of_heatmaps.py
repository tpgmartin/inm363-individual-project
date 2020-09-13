# TODO: Get initial accuracy scores for trained network for original images from test set
import argparse
from glob import glob
import numpy as np
import os
import pandas as pd
from pathlib import Path
import random
import sys
from tcav import utils
import tcav.model as model
import tensorflow as tf

from helpers import make_model, map_labels_to_dirs
from concept_discovery import ConceptDiscovery

def main(args):
    
    sess = utils.create_session()
    mymodel = make_model(sess, args.model_to_run, args.model_path, args.labels_path)

    cd = ConceptDiscovery(
        mymodel,
        args.target_class,
        sess,
        './net_occlusion_heatmaps/n02342885/n02342885_12999/mask_dim_100/n02342885_12999_image_cropped_to_mask/')

    prediction, filename = cd.predict()
    
    sess.close()

    print('Filename:', filename)
    print('True label:', args.target_class)
    print('Predicted label:', mymodel.id_to_label(prediction.tolist()[0].index(np.max(prediction))))
    print('Prediction probability:', np.max(prediction))

def parse_arguments(argv):
  parser = argparse.ArgumentParser()
  parser.add_argument('--source_dir', type=str,
      help='''Directory where the network's classes image folders and random
      concept folders are saved.''', default='../ACE/ImageNet/ILSVRC2012_img_train/')
  parser.add_argument('--model_to_run', type=str,
      help='The name of the model.', default='GoogleNet')
  parser.add_argument('--model_path', type=str,
      help='Path to model checkpoints.', default='./tensorflow_inception_graph.pb')
  parser.add_argument('--labels_path', type=str,
      help='Path to model checkpoints.', default='./imagenet_labels.txt')
  parser.add_argument('--target_class', type=str,
      help='The name of the target class to be interpreted', default='hamster')
  return parser.parse_args(argv)


if __name__ == '__main__':

    main(parse_arguments(sys.argv[1:]))
