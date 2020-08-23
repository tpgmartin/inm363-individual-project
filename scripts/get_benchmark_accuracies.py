# TODO: Get initial accuracy scores for trained network for original images from test set
import argparse
from glob import glob
import numpy as np
import pandas as pd
import random
import sys
from tcav import utils
import tcav.model as model
import tensorflow as tf

from helpers import make_model
from concept_discovery import ConceptDiscovery

def map_labels_to_dirs(labels_to_dirs='./labels/ImageNet_label.txt'):

    labels_to_dirs = open(labels_to_dirs)
    mapping = {}
    for line in labels_to_dirs:
        filename, labels = line.strip().split('    ')
        label = labels.split(', ')[0]
        mapping[label] = filename
    
    return mapping

def main(args):

    mapping_labels_to_dirs = map_labels_to_dirs()
    
    sess = utils.create_session()
    mymodel = make_model(sess, args.model_to_run, args.model_path, args.labels_path)
  
    cd = ConceptDiscovery(
        mymodel,
        mapping_labels_to_dirs[args.target_class],
        sess,
        args.source_dir)

    predictions, filenames = cd.predict()
    
    num_predictions = len(predictions)
    directory = [filenames[0].split('/')[-2]] * num_predictions
    filenames = [filename.split('/')[-1] for _, filename in zip(predictions, filenames)]
    true_labels = [args.target_class] * num_predictions
    predicted_labels = [mymodel.id_to_label(list(prediction).index(np.max(prediction))) for prediction in predictions]
    prediction_probability = [np.max(prediction) for prediction in predictions]

    df = pd.DataFrame({
        'directory': directory,
        'filename': filenames,
        'true_label': true_labels,
        'predicted_label': predicted_labels,
        'prediction_probability': prediction_probability
    })
    
    df.to_csv(f"./data/{'_'.join(args.target_class.split(' '))}_baseline_predictions.csv")


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
      help='The name of the target class to be interpreted', default='coffee mug')
  return parser.parse_args(argv)


if __name__ == '__main__':
    
    random.seed(42)

    args = parse_arguments(sys.argv[1:])
    labels = [label.strip() for label in open(args.labels_path)]
    labels_sample = random.sample(labels, 10)

    for label in labels_sample:
        args.target_class = label
        main(args)
