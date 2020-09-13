from glob import glob
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

    # Occlude Images ######################################################################################################################
    random.seed(42)
    MAX_MASK_SIZE = 100
    MAX_MASKED_IMAGES = 1000
    MASKS_PER_EPOCH = 1

    baseline_prediction_samples = pd.concat([pd.read_csv(f) for f in glob('./baseline_prediction_samples/*')])
    baseline_prediction_samples_filenames = [filename.split('/')[-2] for filename in baseline_prediction_samples['filename'].values]
    existing_occluded_images = [directory.split('/')[-1] for directory in glob('./occluded_images/**/*')]
    images_to_occlude = list(set(baseline_prediction_samples_filenames) - set(existing_occluded_images))

    # images_to_occlude = [image for image in images_to_occlude if 'n02966193_2173' in image]
    for image in images_to_occlude:
        occlude_images.main(image, image.split('_')[0], MAX_MASK_SIZE, MAX_MASKED_IMAGES, MASKS_PER_EPOCH)
    # #####################################################################################################################################

    # Get Occluded Image Accuracy #########################################################################################################
    mapping_images_to_labels = map_images_to_labels()

    keys = ['source_dir', 'model_to_run', 'model_path', 'labels_path', 'target_class']
    values = ['./occluded_images/', 'GoogleNet', './tensorflow_inception_graph.pb', './imagenet_labels.txt', 'bookshop']
    args_params = dict(zip(keys,values))
    args = ArgsDict(args_params)

    # for images_path in [f for f in glob(f'{args.source_dir}**/**/*') if 'n02966193_2173' in f]:
    for images_path in glob(f'{args.source_dir}**/**/*'):
        args.target_class = mapping_images_to_labels[images_path.split('/')[2]]
        args.source_dir = f'{images_path}/'
        get_occluded_image_accuracy.main(args)
    # #####################################################################################################################################

    # Check True Label Prediction Accuracy ################################################################################################
    # for f in ['./occluded_image_predictions/mask_dim_50/bookshop_image_n02871525_10490_occluded_image_predictions.csv']:
    for f in glob('occluded_image_predictions/**/*'):
        check_true_label_prediction_accuracy.main(f'./{f}')
    # #####################################################################################################################################

    # Generate Net Heatmap from Prediction Probabilities ##################################################################################
    occlusion_heatmaps = [f.split('/')[-1] for f in glob('./occluded_image_predictions/**/*')]
    occlusion_heatmaps = [f for f in occlusion_heatmaps if '_' in f]
    existing_heatmaps = [f.split('/')[-1] for f in glob('./net_occlusion_heatmaps/**/*')]
    heatmaps = list(set(occlusion_heatmaps) - set(existing_heatmaps))
    
    # heatmaps = [f for f in occlusion_heatmaps if 'n02966193_2173' in f]
    for f in heatmaps:
        print(f)
        generate_net_heatmap_from_prediction_probabilities.main(('_').join(f.split('_')[2:4]), f.split('_')[0], 1)
    # #####################################################################################################################################