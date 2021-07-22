from glob import glob
import itertools
import pandas as pd
import random

from helpers import map_images_to_labels
import check_true_label_prediction_accuracy
import generate_net_heatmap_from_prediction_probabilities
import get_occluded_image_accuracy
import occlude_images

class ArgsDict(dict):
    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError as k:
            raise AttributeError(k)

    def __setattr__(self, key, value):
        self[key] = value

if __name__ == '__main__':

    argsLists = [
        ['cab'],
        [100],
        [10000],
        [1000],
        [1.644854]
    ]

    target_images = [
        'n02930766_13320',
        'n02930766_14354',
        'n02930766_31200',
        'n02930766_34503',
        'n02930766_23814',
        'n02930766_8716',
        'n02930766_13891',
        'n02930766_7103'
    ]
    
    for label, MAX_MASK_SIZE, MAX_MASKED_IMAGES, STEP, z_val in itertools.product(*argsLists):

        # Occlude Images ######################################################################################################################
        random.seed(42)
        MASKS_PER_EPOCH = 1

        name_lookup = pd.read_csv(f'../ACE/ACE/concepts/images/{label}/image_name_lookup.csv')
        class_id = name_lookup.iloc[0]['image_filename'].split('/')[3]
        print('class_id:', class_id)

        for idx, row in name_lookup.iterrows():

            image_name = row['image_name'].split('.')[0]
            image_filename = row['image_filename']
            image = image_filename.split('/')[-1][:-5]

            if image in target_images:
                occlude_images.main(image, image.split('_')[0], label, image_name, MAX_MASK_SIZE, MAX_MASKED_IMAGES, MASKS_PER_EPOCH)
        # # # #####################################################################################################################################

        # # # Get Occluded Image Accuracy #########################################################################################################
        mapping_images_to_labels = map_images_to_labels()

        keys = ['source_dir', 'model_to_run', 'model_path', 'labels_path', 'target_class']
        values = ['./occluded_images/', 'InceptionV3', './inception_v3.h5', './imagenet_labels.txt', label]
        args_params = dict(zip(keys,values))
        args = ArgsDict(args_params)

        for images_path in [f for f in glob(f'{args.source_dir}**/**/*') if class_id in f]:
            args.target_class = label
            args.source_dir = images_path
            get_occluded_image_accuracy.main(args)
        # # # # #####################################################################################################################################

        # # # # Check True Label Prediction Accuracy ################################################################################################
        for f in [f'./occluded_image_predictions/mask_dim_{MAX_MASK_SIZE}/cinema_image_n03032252_10737_occluded_image_predictions.csv']:
            check_true_label_prediction_accuracy.main(f'./{f}')
        # #####################################################################################################################################

        # Generate Net Heatmap from Prediction Probabilities ##################################################################################
        occlusion_heatmaps = [f.split('/')[-1] for f in glob(f'./occluded_image_predictions/mask_dim_{MAX_MASK_SIZE}/{label}*')]
        existing_heatmaps = [f.split('/')[-1] for f in glob('./net_occlusion_heatmaps_delta_prob/**/*')]
        heatmaps = list(set(occlusion_heatmaps) - set(existing_heatmaps))

        for idx, row in name_lookup.iterrows():

            image_name = row['image_name'].split('.')[0] # == 0001
            image_filename = row['image_filename'] # == ./ImageNet/ILSVRC2012_img_train/n03032252/img_sample/cinema/n03032252_17822.JPEG
            image = image_filename.split('/')[-1][:-5] # == n03032252_17822

            for f in heatmaps:

                if image in target_images:
                    for start in range(0, MAX_MASKED_IMAGES, STEP):
                        generate_net_heatmap_from_prediction_probabilities.main(image, label, image_name, MAX_MASKED_IMAGES, MAX_MASK_SIZE, z_val, start)
        # # #####################################################################################################################################