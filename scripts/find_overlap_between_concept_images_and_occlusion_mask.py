from glob import glob
from multiprocessing import dummy as multiprocessing
import numpy as np
import pandas as pd
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

layer = 'mixed8'
target_class = 'ambulance'
name_lookup = pd.read_csv(f'../ACE/ACE/concepts/images/{target_class}/image_name_lookup.csv')
all_concepts = len(glob(f'../ACE/ACE/masks/{layer}_{target_class}_{target_class}_concept*_masks/'))

images = []
concept_nos = []
img_filenames = []
overlaps = []
for idx, row in name_lookup.iterrows():

    image_name = int(row['image_name'].split('.')[0]) # e.g. 1 - 40
    image_filename = row['image_filename'] # e.g. ./ImageNet/ILSVRC2012_img_train/n03032252/img_sample/cinema/n03032252_17822.JPEG
    image = image_filename.split('/')[-1][:-5] # e.g. n03032252_17822

    if image not in [
        'n02701002_2140',
        'n02701002_3381',
        'n02701002_12971',
        'n02701002_22776',
        'n02701002_4615',
        'n02701002_7033',
        'n02701002_3087',
        'n02701002_2968',
        'n02701002_1415',
        'n02701002_652'
        # 'n02930766_13320',
        # 'n02930766_14354',
        # 'n02930766_31200',
        # 'n02930766_34503',
        # 'n02930766_23814',
        # 'n02930766_8716',
        # 'n02930766_13891',
        # 'n02930766_7103'
    ]:
        continue

    mask_no = image_name - 1
    class_no = image.split('_')[0]
    image_no = image.split('_')[-1]

    print('image:', image)

    # load image segment masks by concept and image
    img_segment_masks = {}
    # TODO: check why class name duplicated
    for concept_no in range(all_concepts):

        concept_no += 1
        img_segment_masks[concept_no] = {}
        for mask_filepath in glob(f'../ACE/ACE/masks/{layer}_{target_class}_{target_class}_concept{concept_no}_masks/*_{mask_no}.npy'):
            mask_filename = mask_filepath.split('/')[-1]

            mask_ = np.load(mask_filepath).squeeze()

            img_segment_masks[concept_no][mask_filename] = np.load(mask_filepath).squeeze()

    # # Load occlusion results
    # ./net_occlusion_heatmaps_delta_prob/n02701002/n02701002_1415/mask_dim_50/n02701002_1415_z_value_1.644854_total_masked_images_1000_image_mask/n02701002_1415_image_mask.npy
    occlusion_img_filename = f'./net_occlusion_heatmaps_delta_prob/{class_no}/{image}/mask_dim_50/{image}_z_value_1.644854_total_masked_images_1000_image_mask/{image}_image_mask.npy'

    # TODO: Handle FileNotFoundError, most images will not have a heatmap image
    occlusion_img_mask = np.load(occlusion_img_filename).squeeze()

    for concept_no in range(all_concepts):

        overlap_sim_by_mask = {}
        max_concept_no = None
        max_img_filename = None
        max_overlap = 0

        concept_no += 1
        img_segment_masks_by_concept = img_segment_masks[concept_no]

        if len(img_segment_masks_by_concept) > 0:
            for img_filename, mask in img_segment_masks_by_concept.items():

                # mask = (...)
                if np.mean(mask) > 0.001:
                    # Update this, as occlusion mask is point of reference only 
                    # care about fraction of mask contained in occluion image
                    # TODO: Need to check if numerator larger than 0 as per denominator
                    # Or for numerator: could take sum of two masks and check total number of elements == 2
                    # overlap = np.sum(occlusion_img_mask * mask) / np.sum(occlusion_img_mask > 0)
                    # overlap = 100 * (np.sum((mask + occlusion_img_mask) > 1) / np.sum(occlusion_img_mask > 0))
                    overlap = 100 * (np.sum((occlusion_img_mask + mask) > 1) / np.sum(mask > 0))
                    print(overlap)
                    
                    if overlap > max_overlap:
                        max_concept_no = concept_no
                        max_img_filename = img_filename
                        max_overlap = overlap
                    # overlap_sim_by_mask[concept_no][img_filename] = overlap
        
        if max_img_filename is not None:
            image = image.split('_')[-1]
            images.append(image)
            concept_nos.append(max_concept_no)
            img_filenames.append(max_img_filename)
            overlaps.append(round(max_overlap,1)) 

df = pd.DataFrame({
    'concept_no': concept_nos,
    'image': images,
    'img_filenames': img_filenames,
    'overlap_percentage': overlaps
})

df.sort_values(by=['concept_no', 'overlap_percentage', 'image'], ascending=[1,0,0], inplace=True)

df.to_csv(f'{layer}_{target_class}_concept_image_segment_overlap.csv', index=False)