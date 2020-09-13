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

from helpers import make_model, map_images_to_labels
from concept_discovery import ConceptDiscovery

def main(args, img_path, true_label):
    
    sess = utils.create_session()
    mymodel = make_model(sess, args.model_to_run, args.model_path, args.labels_path)

    filenames = []
    predictions = []
    baseline_prediction_probs = []

    baseline_predictions = pd.read_csv(f"./baseline_prediction_samples/{true_label}baseline_prediction_samples.csv")

    for img in glob(f'{img_path}/*'):

        cd = ConceptDiscovery(
            mymodel,
            args.target_class,
            sess,
            f"./net_occlusion_heatmaps_delta_prob/{img.split('/')[-1].split('_')[0]}/{img.split('/')[-1]}/mask_dim_100/{img.split('/')[-1]}_image_cropped_to_mask/")

        prediction, filename = cd.predict()

        # No instances where true label != predicted label in sample
        baseline_prediction_probs.append(baseline_predictions[baseline_predictions['filename'].str.contains(img.split('/')[-1])]['prediction_probability'].values[0])
    
        try:
            prediction, filename = cd.predict()
            predictions.append(prediction)
            filenames.append(filename[0])
        except ValueError as e:
            predictions.append(np.nan)
            filenames.append(args.source_dir)
            pass

    sess.close()

    true_labels = [true_label] * len(predictions)
    
    predicted_labels = []
    for prediction in predictions:
        try:
            predicted_labels.append(mymodel.id_to_label(prediction.tolist()[0].index(np.max(prediction))))
        except AttributeError:
            predicted_labels.append(np.nan)

    if args.target_class == 'crane bird':
        args.target_class = 'crane'
    elif args.target_class == 'african grey':
        args.target_class = 'African grey'
    elif args.target_class == 'tank suit':
        args.target_class = 'maillot'

    true_label_predictions = []
    true_label_prediction_delta = []
    for prediction, baseline_prediction_probs in zip(predictions,baseline_prediction_probs):
        try:
            true_label_prediction_prob = prediction.tolist()[0][mymodel.label_to_id(args.target_class)]
            true_label_predictions.append(true_label_prediction_prob)
            true_label_prediction_delta.append(true_label_prediction_prob-baseline_prediction_probs)
        except AttributeError:
            true_label_predictions.append(np.nan)

    prediction_probability = [np.max(prediction) for prediction in predictions]

    df = pd.DataFrame({
        'filename': filenames,
        'true_label': true_labels,
        'true_label_predictions': true_label_predictions,
        'true_label_predictions_delta': true_label_prediction_delta,
        'predicted_label': predicted_labels,
        'prediction_probability': prediction_probability
    })
    
    save_filename = f"./net_heatmap_predictions/mask_dim_100/{'_'.join(true_label.split(' '))}_heatmap_predictions.csv"
    save_filepath = Path(save_filename)
    save_filepath.touch(exist_ok=True)

    df.to_csv(save_filename, index=False)

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
      help='The name of the target class to be interpreted', default='bubble')
  return parser.parse_args(argv)


if __name__ == '__main__':

    mapping = map_images_to_labels()

    for f in [f for f in glob('./net_occlusion_heatmaps_delta_prob/*') if 'n09229709' in f]:
        main(parse_arguments(sys.argv[1:]), f, mapping[f.split('/')[-1]])

