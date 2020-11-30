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

def main(args, activations_dir, cavs_dir, random_concept='random_discovery'):
    
    sess = utils.create_session()
    mymodel = make_model(sess, args.model_to_run, args.model_path, args.labels_path)

    filenames = []
    activations = []
    
    # for source_dir in glob(f'{args.source_dir}/*'):
  
    # cd = ConceptDiscovery(
    #     mymodel,
    #     None,
    #     sess,
    #     f'{args.source_dir}/')

    cd = ConceptDiscovery(
        mymodel,
        args.target_class,
        random_concept,
        args.bottlenecks,
        sess,
        args.source_dir,
        activations_dir,
        cavs_dir)

		# Expect dimensions to be equal to bottleneck layer only - for single image
    bn_activations = cd.get_bn_activations()

    print(len(bn_activations))
    print(len(bn_activations[0]))
    print(bn_activations[0])
    print('~~~~~~~~~~~~~~~~~~~~~~')
        # try:
        #     prediction, filename = cd.predict()
        #     predictions.append(prediction)
        #     filenames.append(filename[0])
        # except ValueError as e:
        #     predictions.append(np.nan)
        #     filenames.append(source_dir)
        #     pass

    sess.close()


    return

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

    # df = pd.DataFrame({
    #     'directory': directory,
    #     'mask_dim': mask_dim,
    #     'filename': filenames,
    #     'true_label': true_labels,
    #     'true_label_predictions': true_label_predictions,
    #     'predicted_label': predicted_labels,
    #     'prediction_probability': prediction_probability
    # })
    
    # save_filename = f"./occluded_image_predictions/mask_dim_{mask_dim[0]}/{'_'.join(args.target_class.split(' '))}_image_{args.source_dir.split('/')[3]}_occluded_image_predictions.csv"
    # save_filepath = Path(save_filename)
    # save_filepath.touch(exist_ok=True)

    # df.to_csv(save_filename, index=False)

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
    # print(glob('./baseline_prediction_samples/*'))

		# TODO: run for all sampled input images
    df = pd.read_csv('./baseline_prediction_samples/cassettebaseline_prediction_samples.csv')
    for idx, row in df.iterrows():
        args.target_class = row['true_label']
        args.source_dir = '/'.join(row['filename'].split('/')[:-1])
        main(args, activations_dir, cavs_dir)
    