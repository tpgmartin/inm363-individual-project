from glob import glob
from multiprocessing import dummy as multiprocessing
import numpy as np
from PIL import Image
import tensorflow as tf

# Need to map input images to image_numbers
# Images in `/ACE/ACE/concepts/images/<label>/*.png` are numbered 0001 - 0040
# Images in `/ACE/ACE/concepts/<layer>_<label>_concept<concept_no>/*.png` are numbered 1 - 
# Latter correspond to image segment masks

# Load image segments
# Load image segments *_39.npy, corresponding to image 10737
# Need to run for each concept and each input image
# ../ACE/ACE/masks/mixed4c_cinema_cinema_concept3_masks/*_.npy

layer = 'mixed4c'
target_class = 'cinema'
all_concepts = len(glob(f'../ACE/ACE/masks/{layer}_{target_class}_{target_class}_concept*_masks/'))

# load image segment masks by concept and image
img_segment_masks = {}
# TODO: check why class name duplicated
for concept_no in range(all_concepts):

    concept_no += 1
    img_segment_masks[concept_no] = {}
    for mask_filepath in glob(f'../ACE/ACE/masks/{layer}_{target_class}_{target_class}_concept{concept_no}_masks/*_39.npy'):
        mask_filename = mask_filepath.split('/')[-1]

        mask_ = np.load(mask_filepath).squeeze()

        img_segment_masks[concept_no][mask_filename] = np.load(mask_filepath).squeeze()


# # Load occlusion results
occlusion_img_filename = './net_occlusion_heatmaps_delta_prob/n03032252/n03032252_10737/mask_dim_100/n03032252_10737_image_mask/n03032252_10737_image_mask.npy'
occlusion_img_mask = np.load(occlusion_img_filename).squeeze()

jaccard_sim_by_mask = {}
max_concept_no = None
max_img_filename = None
max_jaccard = 0
for concept_no in range(all_concepts):

    concept_no += 1
    img_segment_masks_by_concept = img_segment_masks[concept_no]

    if len(img_segment_masks_by_concept) > 0:
        for img_filename, mask in img_segment_masks_by_concept.items():

            # mask = (...)
            if np.mean(mask) > 0.001:
                jaccard = np.sum(occlusion_img_mask * mask) / np.sum((occlusion_img_mask + mask) > 0)
                
                if jaccard > max_jaccard:
                    max_concept_no = concept_no
                    max_img_filename = img_filename
                    max_jaccard = jaccard
                # jaccard_sim_by_mask[concept_no][img_filename] = jaccard

print(concept_no)
print(img_filename)
print(jaccard)
print('----------')