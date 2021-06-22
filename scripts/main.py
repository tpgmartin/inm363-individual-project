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
    MAX_MASK_SIZE = 50
    MAX_MASKED_IMAGES = 100
    MASKS_PER_EPOCH = 1
    label = 'cinema'

    name_lookup = pd.read_csv(f'../ACE/ACE/concepts/images/{label}/image_name_lookup.csv')
    class_id = name_lookup.iloc[0]['image_filename'].split('/')[3]
    print('class_id:', class_id)

    # for idx, row in name_lookup.iterrows():

    #     image_name = row['image_name'].split('.')[0]
    #     image_filename = row['image_filename']
    #     image = image_filename.split('/')[-1][:-5]

    #     occlude_images.main(image, image.split('_')[0], label, image_name, MAX_MASK_SIZE, MAX_MASKED_IMAGES, MASKS_PER_EPOCH)
    # # # #####################################################################################################################################

    # # # Get Occluded Image Accuracy #########################################################################################################
    mapping_images_to_labels = map_images_to_labels()

    keys = ['source_dir', 'model_to_run', 'model_path', 'labels_path', 'target_class']
    values = ['./occluded_images/', 'GoogleNet', './tensorflow_inception_graph.pb', './imagenet_labels.txt', label]
    args_params = dict(zip(keys,values))
    args = ArgsDict(args_params)

    for images_path in [f for f in glob(f'{args.source_dir}**/**/*') if class_id in f]:
    # for images_path in glob(f'{args.source_dir}**/**/*'):
    # for images_path in ['./occluded_images/{class_id}/n03032252_10737/mask_dim_100']:
        args.target_class = mapping_images_to_labels[images_path.split('/')[2]]
        args.source_dir = f'{images_path}/'
        get_occluded_image_accuracy.main(args)
    # # # #####################################################################################################################################

    # # # Check True Label Prediction Accuracy ################################################################################################
    for f in glob(f'./occluded_image_predictions/mask_dim_{MAX_MASK_SIZE}/{label}_image_*.csv'):
    # # # for f in glob('occluded_image_predictions/**/*'):
    # for f in [f'./occluded_image_predictions/mask_dim_{MAX_MASK_SIZE}/cinema_image_n03032252_10737_occluded_image_predictions.csv']:
        check_true_label_prediction_accuracy.main(f'./{f}')
    # #####################################################################################################################################

    # Generate Net Heatmap from Prediction Probabilities ##################################################################################
    occlusion_heatmaps = [f.split('/')[-1] for f in glob(f'./occluded_image_predictions/mask_dim_{MAX_MASK_SIZE}/{label}*')]
    # occlusion_heatmaps = [f for f in occlusion_heatmaps if '_' in f]
    existing_heatmaps = [f.split('/')[-1] for f in glob('./net_occlusion_heatmaps_delta_prob/**/*')]
    # existing_heatmaps.append('jeep_image_n03594945_13257_occluded_image_predictions.csv')
    heatmaps = list(set(occlusion_heatmaps) - set(existing_heatmaps))

    # heatmaps = [
    #     'cinema_image_n03032252_10737_occluded_image_predictions.csv'
    # ]
    
    # heatmaps = [f for f in occlusion_heatmaps if class_id in f]

    for idx, row in name_lookup.iterrows():

        image_name = row['image_name'].split('.')[0] # == 0001
        image_filename = row['image_filename'] # == ./ImageNet/ILSVRC2012_img_train/n03032252/img_sample/cinema/n03032252_17822.JPEG
        image = image_filename.split('/')[-1][:-5] # == n03032252_17822

    # for f in heatmaps:
        # print(f)
        # f == cinema_image_n03032252_50142_occluded_image_predictions.csv
        # ('_').join(f.split('_')[2:4]) == n03032252_10737
        # f.split('_')[0] == cinema
        # generate_net_heatmap_from_prediction_probabilities.main(('_').join(f.split('_')[2:4]), f.split('_')[0], 'cinema', '0040', MAX_MASK_SIZE, 1.644854)
        generate_net_heatmap_from_prediction_probabilities.main(image, 'cinema', 'cinema', image_name, MAX_MASK_SIZE, 1.644854)
    # #####################################################################################################################################