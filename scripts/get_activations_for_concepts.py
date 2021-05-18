import argparse
from glob import glob
import numpy as np
import os
import pandas as pd
from pathlib import Path
import random
from shutil import copyfile
import sys
from tcav import utils
import tcav.model as model
import tensorflow as tf
import time

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

		bn_activations = cd.get_img_activations(args.img_num, args.concept_num)

		sess.close()

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
			help='The name of the target class to be interpreted', default='ambulance')
		parser.add_argument('--bottlenecks', type=str,
				help='Names of the target layers of the network (comma separated)',
												default='mixed4c')
		parser.add_argument('--num_random_exp', type=int,
				help="Number of random experiments used for statistical testing, etc",
												default=20)
		return parser.parse_args(argv)


if __name__ == '__main__':

		main_start = time.time()

		layer = 'mixed4c'
		args = parse_arguments(sys.argv[1:])
		# args.model_to_run = 'InceptionV3'
		# args.model_path = './inception_v3.h5'
		args.bottlenecks = layer
		random_concept = 'random_discovery'
		cavs_dir = os.path.join(args.working_dir, 'cavs/')
		activations_dir = os.path.join(args.working_dir, 'acts/')
		tf.gfile.MakeDirs(cavs_dir)
		tf.gfile.MakeDirs(activations_dir)
		
		samples = glob('./baseline_prediction_samples/*')

		l = []
		target_labels = [
			'bullet_train',
			'police_van',
			'jeep',
			'mantis'
		]
		# 'restaurant', 'cinema', 'cab', 'bookshop', 'ambulance', 
		#  'lipstick', 'lotion', 'volleyball', 'basketball', 'ant', 
		#  'mantis', 'snail', 'damselfly', 'bubble', 'balloon']

		# for img in l:

		# Get random images

		# random_sample = random.sample(glob('../ACE/ImageNet/random500_0/*.JPEG'),40)

		for label in target_labels:
			print(label, '-----------------------')
			chart_type = 'top_10'
			start = 4
			stop = start + int(chart_type.split('_')[-1])

			with open(f'../ACE/ACE/results_summaries/{layer}_{label}_ace_results.txt') as f:
				lines = f.readlines()

			top_concepts = [line.split(':')[1].split('_')[-1] for line in lines[start:stop]]
			
			concept_imgs = [x for x in glob(f'../ACE/ACE/concepts/{layer}_{label}_*/*.png') if 'patches' not in x]
			concept_imgs = [concept_img for concept_img in concept_imgs if concept_img.split('/')[-2].split('_')[-1] in top_concepts]

			activations = glob(f'./acts/{label}/acts_{label}_concept*_*_{layer}')
			activation_concept_imgs = []
			for activation in activations:
				l = activation.split('_')
				concept, image_no_1, image_no_2 = l[2], l[3], l[4]
				image_no = f'{image_no_1}_{image_no_2}'
				activation_concept_img = f'../ACE/ACE/concepts/{layer}_{label}_{concept}/{image_no}.png'
				activation_concept_imgs.append(activation_concept_img)

			concept_imgs = list(set(concept_imgs) - set(activation_concept_imgs))

			# concept21_imgs = [x for x in concept_imgs if 'concept21' in x]
			# concept2_imgs = [x for x in concept_imgs if 'concept2' in x]
			# concept13_imgs = [x for x in concept_imgs if 'concept13' in x]
			# concept9_imgs = [x for x in concept_imgs if 'concept9' in x]

			# concept_imgs = concept21_imgs + concept2_imgs + concept13_imgs + concept9_imgs
			print(concept_imgs)
			for concept_img in concept_imgs:
			# for concept_img in random_sample:
				img_start = time.time()
				concept_num = 'random' if concept_img.split('/')[-2].split('_')[-1] == '0' else concept_img.split('/')[-2].split('_')[-1]
				true_label = 'random' if concept_img.split('/')[-2].split('_')[1] == '0' else concept_img.split('/')[-2].split('_')[1]
				source_dir = concept_img[:-4]
				img_filename = concept_img.split('/')[-1].split('.')[0]
				# This step creates a new folder for each image file to pass to model
				source_file = f'{source_dir}/{img_filename}.png'

				os.makedirs(source_dir, exist_ok=True)
				copyfile(concept_img, source_file)

				args.target_class = true_label
				args.source_dir = source_dir
				args.img_num = img_filename
				args.concept_num = concept_num
				bottleneck_activation = main(args, activations_dir, cavs_dir)
				img_end = time.time()
				print(f'Img elapsed time (s): {img_end - img_start}')
			
			main_end = time.time()
			print(f'Total elapsed time (s): {main_end - main_start}')
			print('End of label ----------')
