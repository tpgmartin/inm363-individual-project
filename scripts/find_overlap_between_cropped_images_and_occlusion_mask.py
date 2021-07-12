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
target_class = 'cab'
name_lookup = pd.read_csv(f'../ACE/ACE/concepts/images/{target_class}/image_name_lookup.csv')

images = []
image_nums = []
overlaps = []

for idx, row in name_lookup.iterrows():

    image_file_num = row['image_name'].split('.')[0]
    image_name = int(row['image_name'].split('.')[0]) # e.g. 1 - 40
    image_filename = row['image_filename'] # e.g. ./ImageNet/ILSVRC2012_img_train/n03032252/img_sample/cinema/n03032252_17822.JPEG
    image = image_filename.split('/')[-1][:-5] # e.g. n03032252_17822

    if image not in [
        'n02930766_13320',
        'n02930766_14354',
        'n02930766_31200',
        'n02930766_34503',
        'n02930766_23814',
        'n02930766_8716',
        'n02930766_13891',
        'n02930766_7103'
    ]:
        continue

    class_no = image.split('_')[0]
    image_no = image.split('_')[-1]

    # load image segment masks by concept and image
    occlusion_mask_overlap = {}

    # # Load occlusion results
    # ./net_occlusion_heatmaps_delta_prob/n02701002/n02701002_1415/mask_dim_50/n02701002_1415_z_value_1.644854_total_masked_images_1000_image_mask/n02701002_1415_image_mask.npy
    occlusion_img_filename = f'./net_occlusion_heatmaps_delta_prob/{class_no}/{image}/mask_dim_50/{image}_z_value_1.644854_total_masked_images_1000_image_mask/{image}_image_mask.npy'
    # TODO: Handle FileNotFoundError, most images will not have a heatmap image
    occlusion_img_mask = np.load(occlusion_img_filename).squeeze()

    '/Users/thomasmartin/Documents/_repos/yolov5/runs/detect/exp/crops'
    crop_img_mask = np.load(f'../yolov5/runs/detect/exp/crops/{image_file_num}.npy').squeeze()
    # print('crop_img_mask')
    # for y in range(len(crop_img_mask)):
    #     for x in range(len(crop_img_mask[0])):
    #         if crop_img_mask[y][x] != 0:
    #             print(crop_img_mask[y][x])
    # print('occlusion_img_mask')
    # for y in range(len(occlusion_img_mask)):
    #     for x in range(len(occlusion_img_mask[0])):
    #         if occlusion_img_mask[y][x] != 0:
    #             print(occlusion_img_mask[y][x])
    # print(np.sum((crop_img_mask + occlusion_img_mask) > 0))
    # print(np.sum(crop_img_mask > 0))
    # print(np.sum(occlusion_img_mask > 0))
    overlap = 100 * (np.sum((crop_img_mask + occlusion_img_mask) > 1) / np.sum(crop_img_mask > 0))

    print('image:', image)
    print(overlap) 
    print()
        
    images.append(image)
    image_nums.append(image_file_num)
    overlaps.append(round(overlap,1)) 

df = pd.DataFrame({
    'image': images,
    'image_num': image_nums,
    'overlap_percentage': overlaps
})

print(df)

# df.to_csv(f'{layer}_{target_class}_occlusion_image_overlap_with_crop_image.csv', index=False)