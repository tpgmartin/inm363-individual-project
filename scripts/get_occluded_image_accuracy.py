import argparse
from glob import glob
import numpy as np
import pandas as pd
from pathlib import Path
import random
import sys
from tcav import utils
import tcav.model as model
import tensorflow as tf

from helpers import make_model, map_images_to_labels
from concept_discovery import ConceptDiscovery

def main(args):
    
    sess = utils.create_session()
    mymodel = make_model(sess, args.model_to_run, args.model_path, args.labels_path)

    filenames = []
    predictions = []
    
    for source_dir in glob(f'{args.source_dir}/*'):
  
        cd = ConceptDiscovery(
            mymodel,
            None,
            None,
            None,
            sess,
            f'{source_dir}/',
            None,
            None)

        try:
            prediction, filename = cd.predict()
            predictions.append(prediction)
            filenames.append(filename[0])
        except ValueError as e:
            predictions.append(np.nan)
            filenames.append(source_dir)
            pass

    sess.close()

    num_predictions = len(predictions)
    directory = ['/'.join(args.source_dir.split('/')[:4])] * num_predictions
    mask_dim = [args.source_dir.split('/')[-2].split('_')[-1]] * num_predictions
    true_labels = [args.target_class] * num_predictions

    # For occluded image prediction need to find prediction accuracy of true label
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
    for prediction in predictions:
        try:
            true_label_predictions.append(prediction.tolist()[0][mymodel.label_to_id(args.target_class)])
        except AttributeError:
            true_label_predictions.append(np.nan)

    prediction_probability = [np.max(prediction) for prediction in predictions]

    df = pd.DataFrame({
        'directory': directory,
        'mask_dim': mask_dim,
        'filename': filenames,
        'true_label': true_labels,
        'true_label_predictions': true_label_predictions,
        'predicted_label': predicted_labels,
        'prediction_probability': prediction_probability
    })
    
    # save_filename = f"./occluded_image_predictions/mask_dim_{mask_dim[0]}/{'_'.join(args.target_class.split(' '))}_image_{args.source_dir.split('/')[3]}_occluded_image_predictions.csv"
    save_filename = 'occluded_image_predictions/mask_dim_100/cab_image_n02930766_23814_occluded_image_predictions.csv'
    save_filepath = Path(save_filename)
    save_filepath.touch(exist_ok=True)

    df.to_csv(save_filename, index=False)

def parse_arguments(argv):
  parser = argparse.ArgumentParser()
  parser.add_argument('--source_dir', type=str,
      help='''Directory where the network's classes image folders and random
      concept folders are saved.''', default='./occluded_images/')
  parser.add_argument('--model_to_run', type=str,
      help='The name of the model.', default='GoogleNet')
  parser.add_argument('--model_path', type=str,
      help='Path to model checkpoints.', default='./tensorflow_inception_graph.pb')
  parser.add_argument('--labels_path', type=str,
      help='Path to model checkpoints.', default='./imagenet_labels.txt')
  parser.add_argument('--target_class', type=str,
      help='The name of the target class to be interpreted', default='bookshop')
  return parser.parse_args(argv)


if __name__ == '__main__':

    mapping_images_to_labels = map_images_to_labels()
    args = parse_arguments(sys.argv[1:])

    for images_path in [f for f in glob(f'{args.source_dir}**/**/*') if 'n09229709' in f]:
        args.target_class = mapping_images_to_labels[images_path.split('/')[2]]
        args.source_dir = f'{images_path}/'
        main(args)
    