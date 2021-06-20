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

layer = 'mixed4c'
target_class = 'cinema'
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
    occlusion_img_filename = f'./net_occlusion_heatmaps_delta_prob/{class_no}/{image}/mask_dim_50/{image}_image_mask/{image}_image_mask.npy'
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
                    overlap = np.sum((mask + occlusion_img_mask) > 1) / np.sum(occlusion_img_mask > 0)
                    
                    if overlap > max_overlap:
                        max_concept_no = concept_no
                        max_img_filename = img_filename
                        max_overlap = overlap
                    # overlap_sim_by_mask[concept_no][img_filename] = overlap
        
        images.append(image)
        concept_nos.append(max_concept_no)
        img_filenames.append(max_img_filename)
        overlaps.append(max_overlap)

df = pd.DataFrame({
    'images': images,
    'concept_nos': concept_nos,
    'img_filenames': img_filenames,
    'overlaps': overlaps
})

df.to_csv(f'{layer}_{target_class}_concept_image_segment_overlap.csv', index=False)