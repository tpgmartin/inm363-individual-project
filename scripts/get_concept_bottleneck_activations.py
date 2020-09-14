import argparse
from glob import glob
import numpy as np
import os
import pandas as pd
from pathlib import Path
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

    main(parse_arguments(sys.argv[1:]))