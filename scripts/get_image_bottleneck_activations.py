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


		imgs = [
			'./baseline_prediction_samples/padlockbaseline_prediction_samples.csv',
			'./baseline_prediction_samples/restaurantbaseline_prediction_samples.csv',
			'./baseline_prediction_samples/hen-of-the-woodsbaseline_prediction_samples.csv',
			'./baseline_prediction_samples/hippopotamusbaseline_prediction_samples.csv',
			'./baseline_prediction_samples/volcanobaseline_prediction_samples.csv',
			'./baseline_prediction_samples/miniskirtbaseline_prediction_samples.csv',
			'./baseline_prediction_samples/volleyballbaseline_prediction_samples.csv',
			'./baseline_prediction_samples/standard_schnauzerbaseline_prediction_samples.csv',
			'./baseline_prediction_samples/lionfishbaseline_prediction_samples.csv',
			'./baseline_prediction_samples/tripodbaseline_prediction_samples.csv',
			'./baseline_prediction_samples/tiger_beetlebaseline_prediction_samples.csv',
			'./baseline_prediction_samples/thresherbaseline_prediction_samples.csv',
			'./baseline_prediction_samples/butcher_shopbaseline_prediction_samples.csv',
			'./baseline_prediction_samples/crossword_puzzlebaseline_prediction_samples.csv',
			'./baseline_prediction_samples/ambulancebaseline_prediction_samples.csv',
			'./baseline_prediction_samples/lipstickbaseline_prediction_samples.csv',
			'./baseline_prediction_samples/electric_guitarbaseline_prediction_samples.csv',
			'./baseline_prediction_samples/beer_bottlebaseline_prediction_samples.csv',
			'./baseline_prediction_samples/bullet_trainbaseline_prediction_samples.csv',
			'./baseline_prediction_samples/king_snakebaseline_prediction_samples.csv',
			'./baseline_prediction_samples/balloonbaseline_prediction_samples.csv',
			'./baseline_prediction_samples/boxerbaseline_prediction_samples.csv',
			'./baseline_prediction_samples/dambaseline_prediction_samples.csv',
			'./baseline_prediction_samples/park_benchbaseline_prediction_samples.csv',
			'./baseline_prediction_samples/suitbaseline_prediction_samples.csv',
			'./baseline_prediction_samples/carouselbaseline_prediction_samples.csv',
			'./baseline_prediction_samples/crane_birdbaseline_prediction_samples.csv',
			'./baseline_prediction_samples/leopardbaseline_prediction_samples.csv',
			'./baseline_prediction_samples/American_black_bearbaseline_prediction_samples.csv',
			'./baseline_prediction_samples/wine_bottlebaseline_prediction_samples.csv',
			'./baseline_prediction_samples/Chihuahuabaseline_prediction_samples.csv',
			'./baseline_prediction_samples/cinemabaseline_prediction_samples.csv',
			'./baseline_prediction_samples/panpipebaseline_prediction_samples.csv',
			'./baseline_prediction_samples/gownbaseline_prediction_samples.csv',
			'./baseline_prediction_samples/revolverbaseline_prediction_samples.csv',
			'./baseline_prediction_samples/bell_pepperbaseline_prediction_samples.csv',
			'./baseline_prediction_samples/chowbaseline_prediction_samples.csv',
			'./baseline_prediction_samples/bookshopbaseline_prediction_samples.csv',
			'./baseline_prediction_samples/bassetbaseline_prediction_samples.csv',
			'./baseline_prediction_samples/drilling_platformbaseline_prediction_samples.csv',
			'./baseline_prediction_samples/hard_discbaseline_prediction_samples.csv',
			'./baseline_prediction_samples/reelbaseline_prediction_samples.csv',
			'./baseline_prediction_samples/shopping_cartbaseline_prediction_samples.csv',
			'./baseline_prediction_samples/hatchetbaseline_prediction_samples.csv',
			'./baseline_prediction_samples/damselflybaseline_prediction_samples.csv',
			'./baseline_prediction_samples/woolbaseline_prediction_samples.csv',
			'./baseline_prediction_samples/strainerbaseline_prediction_samples.csv',
			'./baseline_prediction_samples/prisonbaseline_prediction_samples.csv',
			'./baseline_prediction_samples/snailbaseline_prediction_samples.csv',
			'./baseline_prediction_samples/cabbaseline_prediction_samples.csv',
			'./baseline_prediction_samples/linerbaseline_prediction_samples.csv',
			'./baseline_prediction_samples/wood_rabbitbaseline_prediction_samples.csv',
			'./baseline_prediction_samples/traybaseline_prediction_samples.csv',
			'./baseline_prediction_samples/pickelhaubebaseline_prediction_samples.csv',
			'./baseline_prediction_samples/colliebaseline_prediction_samples.csv',
			'./baseline_prediction_samples/theater_curtainbaseline_prediction_samples.csv',
			'./baseline_prediction_samples/mantisbaseline_prediction_samples.csv',
			'./baseline_prediction_samples/Norfolk_terrierbaseline_prediction_samples.csv',
			'./baseline_prediction_samples/killer_whalebaseline_prediction_samples.csv',
			'./baseline_prediction_samples/dumbbellbaseline_prediction_samples.csv',
			'./baseline_prediction_samples/bolo_tiebaseline_prediction_samples.csv',
			'./baseline_prediction_samples/punching_bagbaseline_prediction_samples.csv',
			'./baseline_prediction_samples/llamabaseline_prediction_samples.csv',
			'./baseline_prediction_samples/snorkelbaseline_prediction_samples.csv',
			'./baseline_prediction_samples/toilet_tissuebaseline_prediction_samples.csv',
			'./baseline_prediction_samples/backpackbaseline_prediction_samples.csv',
			'./baseline_prediction_samples/hamsterbaseline_prediction_samples.csv',
			'./baseline_prediction_samples/school_busbaseline_prediction_samples.csv',
			'./baseline_prediction_samples/lotionbaseline_prediction_samples.csv',
			'./baseline_prediction_samples/antbaseline_prediction_samples.csv',
			'./baseline_prediction_samples/jeepbaseline_prediction_samples.csv',
			'./baseline_prediction_samples/Staffordshire_bullterrierbaseline_prediction_samples.csv',
			'./baseline_prediction_samples/European_gallinulebaseline_prediction_samples.csv',
			'./baseline_prediction_samples/marimbabaseline_prediction_samples.csv',
			'./baseline_prediction_samples/gorillabaseline_prediction_samples.csv',
			'./baseline_prediction_samples/Sussex_spanielbaseline_prediction_samples.csv',
			'./baseline_prediction_samples/police_vanbaseline_prediction_samples.csv',
			'./baseline_prediction_samples/china_cabinetbaseline_prediction_samples.csv',
			'./baseline_prediction_samples/tennis_ballbaseline_prediction_samples.csv',
			'./baseline_prediction_samples/space_barbaseline_prediction_samples.csv',
			'./baseline_prediction_samples/african_greybaseline_prediction_samples.csv',
			'./baseline_prediction_samples/zebrabaseline_prediction_samples.csv',
			'./baseline_prediction_samples/jackfruitbaseline_prediction_samples.csv',
			'./baseline_prediction_samples/moving_vanbaseline_prediction_samples.csv',
			'./baseline_prediction_samples/bubblebaseline_prediction_samples.csv',
			'./baseline_prediction_samples/common_newtbaseline_prediction_samples.csv',
			'./baseline_prediction_samples/tank_suitbaseline_prediction_samples.csv',
			'./baseline_prediction_samples/jinrikishabaseline_prediction_samples.csv',
			'./baseline_prediction_samples/sunglassesbaseline_prediction_samples.csv',
			'./baseline_prediction_samples/seashorebaseline_prediction_samples.csv',
			'./baseline_prediction_samples/juncobaseline_prediction_samples.csv',
			'./baseline_prediction_samples/cocker_spanielbaseline_prediction_samples.csv',
			'./baseline_prediction_samples/traffic_lightbaseline_prediction_samples.csv',
			'./baseline_prediction_samples/saxbaseline_prediction_samples.csv',
			'./baseline_prediction_samples/basketballbaseline_prediction_samples.csv',
			'./baseline_prediction_samples/cassettebaseline_prediction_samples.csv'
		]

		# for img in glob('./baseline_prediction_samples/*'):
		for img in imgs:

			df = pd.read_csv(img)

			filenames = []
			bottlneck_layers = []
			true_labels = []
			bottleneck_activations = []
			for idx, row in df.iterrows():
				
				true_label = row['true_label']
				filename = row['filename']

				args.target_class = true_label
				args.source_dir = '/'.join(filename.split('/')[:-1])
				bottleneck_activation = main(args, activations_dir, cavs_dir)

				filenames.append(filename)
				bottlneck_layers.append(args.bottlenecks)
				true_labels.append(true_label)
				bottleneck_activations.append(bottleneck_activation)

			df = pd.DataFrame({
				'filename': filenames,
				'bottleneck_layer': bottlneck_layers,
				'true_label': true_labels,
				'bottleneck_activation': bottleneck_activations
			})

			df.to_csv(f'./bottleneck_activations/{true_labels[0]}_bottleneck_activations.csv', index=False)
