# Script started from https://github.com/amiratag/ACE/blob/c70d605750ef16df98e270e94a00f92d3b070bd5/ace_run.py

# TODO
# * Select each image for each selected label (in `acts` directory)
# * Pass image to ConceptDiscovery instance
# * Find CAV vectors and save to `pkl` file, use format `bubble_concept0-random500_0-mixed4c-linear-0.01`
# * For each CAV find bottleneck activation

import argparse
from glob import glob
import numpy as np
import os
import pandas as pd
import shutil
import sklearn.metrics as metrics
import sys
from tcav import utils
import tensorflow as tf

import helpers
from concept_discovery import ConceptDiscovery


def main(args):

  discovered_concepts_dir = os.path.join(args.working_dir, 'concepts/')
  results_dir = os.path.join(args.working_dir, 'results/')
  cavs_dir = os.path.join(args.working_dir, 'cavs/')
  activations_dir = os.path.join(args.working_dir, 'acts/')
  results_summaries_dir = os.path.join(args.working_dir, 'results_summaries/')

  tf.gfile.MakeDirs(discovered_concepts_dir)
  tf.gfile.MakeDirs(results_dir)
  tf.gfile.MakeDirs(results_summaries_dir)

  random_concept = 'random_discovery'  # Random concept for statistical testing
  sess = utils.create_session()
  mymodel = helpers.make_model(
      sess, args.model_to_run, args.model_path, args.labels_path)
  # Creating the ConceptDiscovery class instance
  cd = ConceptDiscovery(
      mymodel,
      args.target_class,
      random_concept,
      args.bottlenecks.split(','),
      sess,
      args.source_dir,
      activations_dir,
      cavs_dir,
      num_random_exp=args.num_random_exp,
      channel_mean=True,
      max_imgs=args.max_imgs,
      min_imgs=args.min_imgs,
      num_discovery_imgs=args.max_imgs,
      num_workers=args.num_parallel_workers)
  # Creating the dataset of image patches
  cd.create_patches(param_dict={'n_segments': [15, 50, 80]})
  # Saving the concept discovery target class images
  image_dir = os.path.join(discovered_concepts_dir, 'images')
  tf.gfile.MakeDirs(image_dir)
  helpers.save_images(image_dir,
                            (cd.discovery_images * 256).astype(np.uint8))
  # Discovering Concepts
  cd.discover_concepts(method='KM', param_dicts={'n_clusters': 25})
  del cd.dataset  # Free memory
  del cd.image_numbers
  del cd.patches
  # Save discovered concept images (resized and original sized)
  helpers.save_concepts(cd, discovered_concepts_dir)
  # Delete concepts that don't pass statistical testing
  cd.test_and_remove_concepts(scores)

def parse_arguments(argv):
  """Parses the arguments passed to the run.py script."""
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
  parser.add_argument('--max_imgs', type=int,
      help="Maximum number of images in a discovered concept",
                      default=40)
  parser.add_argument('--min_imgs', type=int,
      help="Minimum number of images in a discovered concept",
                      default=40)
  parser.add_argument('--num_parallel_workers', type=int,
      help="Number of parallel jobs.",
                      default=0)
  return parser.parse_args(argv)


if __name__ == '__main__':

    samples = [
        # './baseline_prediction_samples/mantisbaseline_prediction_samples.csv',
        # './baseline_prediction_samples/antbaseline_prediction_samples.csv',
        # './baseline_prediction_samples/lipstickbaseline_prediction_samples.csv',
        # './baseline_prediction_samples/jeepbaseline_prediction_samples.csv',
        # './baseline_prediction_samples/restaurantbaseline_prediction_samples.csv',
        # './baseline_prediction_samples/basketballbaseline_prediction_samples.csv', 
        # './baseline_prediction_samples/bookshopbaseline_prediction_samples.csv',
        # './baseline_prediction_samples/snailbaseline_prediction_samples.csv',
        # './baseline_prediction_samples/damselflybaseline_prediction_samples.csv',
        # './baseline_prediction_samples/lotionbaseline_prediction_samples.csv',
        # './baseline_prediction_samples/bubblebaseline_prediction_samples.csv',
        # './baseline_prediction_samples/cinemabaseline_prediction_samples.csv',
        # './baseline_prediction_samples/ambulancebaseline_prediction_samples.csv',
        './baseline_prediction_samples/balloonbaseline_prediction_samples.csv',
        # './baseline_prediction_samples/cabbaseline_prediction_samples.csv',
        # './baseline_prediction_samples/volleyballbaseline_prediction_samples.csv'
    ]

    for sample in samples:

        df = pd.read_csv(sample)
        filepaths = df['filename']
        sample_dir_path = '/'.join(filepaths[0].split('/')[:-2])
        sample_dir_path += '/img_sample'

        if not os.path.isdir(sample_dir_path):
            os.mkdir(sample_dir_path)

        for f in filepaths:
            shutil.copy(f, sample_dir_path)

        args = parse_arguments(sys.argv[1:])
        args.source_dir = sample_dir_path
        args.target_class = filepaths[0].split('/')[-1].split('baseline')[0]
        main(parse_arguments(sys.argv[1:]))

    # TODO:
    # * Dynamically update true label
    # * Dynamically update image filepath for each image in label subset
