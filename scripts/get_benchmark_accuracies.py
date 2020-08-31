# TODO: Get initial accuracy scores for trained network for original images from test set
import argparse
from glob import glob
import numpy as np
import os
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
    source_dirs = f'{args.source_dir}{mapping_labels_to_dirs[args.target_class]}'

    prediction_count = 0
    filenames = []
    predictions = []

    print(f'Starting benchmark accuracies for {args.target_class}')
    for source_dir in glob(f'{source_dirs}/*'):

        cd = ConceptDiscovery(
            mymodel,
            mapping_labels_to_dirs[args.target_class],
            sess,
            f'{source_dir}/')

        try:
            prediction, filename = cd.predict()
            predictions.append(prediction)
            filenames.append(filename[0])
            prediction_count += 1
        except ValueError as e:
            predictions.append(np.nan)
            filenames.append(source_dir)
            pass
    
    num_predictions = len(predictions)
    directory = [source_dirs.split('/')[-2]] * num_predictions
    true_labels = [args.target_class] * num_predictions

    predicted_labels = []
    for prediction in predictions:
        try:
            predicted_labels.append(mymodel.id_to_label(prediction.tolist()[0].index(np.max(prediction))))
        except AttributeError:
            predicted_labels.append(np.nan)

    prediction_probability = [np.max(prediction) for prediction in predictions]

    df = pd.DataFrame({
        'directory': directory,
        'filename': filenames,
        'true_label': true_labels,
        'predicted_label': predicted_labels,
        'prediction_probability': prediction_probability
    })
    
    df.to_csv(f"./baseline_predictions/{'_'.join(args.target_class.split(' '))}_baseline_predictions.csv")
    print(f'Finished benchmark accuracies for {args.target_class}')


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
      help='The name of the target class to be interpreted', default='tench')
  return parser.parse_args(argv)


if __name__ == '__main__':
    
    random.seed(42)

    args = parse_arguments(sys.argv[1:])
    labels = [label.strip() for label in open(args.labels_path)]

    for label in labels:
        args.target_class = label
        main(args)

    print('End of script!!!')