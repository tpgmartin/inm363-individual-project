from glob import glob
import itertools
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import random
from skimage import draw
from skimage import io
import time

from helpers import map_labels_to_dirs

def get_mask_coords(CURRENT_SEARCHES, previous_mask_coords, HEIGHT, WIDTH, MASK_SIZE, MAX_MASKS_PER_EPOCH):

    MASK_OVERLAP_PREVIOUS = False
    mask_y_min = random.randint(0,HEIGHT-MASK_SIZE)
    mask_x_min = random.randint(0,WIDTH-MASK_SIZE)

    mask_y_coords = [i for i in range(mask_y_min,mask_y_min+MASK_SIZE)]
    mask_x_coords = [i for i in range(mask_x_min,mask_x_min+MASK_SIZE)]
    mask_coords = [coord for coord in itertools.product(*[mask_y_coords,mask_x_coords])]

    for (y, x) in mask_coords:
        if (y, x) in previous_mask_coords:
            MASK_OVERLAP_PREVIOUS = True
            break

    if MASK_OVERLAP_PREVIOUS == True:
        CURRENT_SEARCHES += 1
        if CURRENT_SEARCHES < MAX_MASKS_PER_EPOCH:
            return get_mask_coords(CURRENT_SEARCHES, previous_mask_coords, HEIGHT, WIDTH, MASK_SIZE, MAX_MASKS_PER_EPOCH)
        else:
            return None
    else:
        previous_mask_coords.extend(mask_coords)
        return (mask_y_min, mask_x_min)

def main(img_filename, dir_name, label, img_file, MAX_MASK_SIZE, MAX_MASKED_IMAGES, MASKS_PER_EPOCH):


    # IMG = io.imread(f'../ACE/ImageNet/ILSVRC2012_img_train/{dir_name}/{img_filename}/{img_filename}.JPEG')
    IMG = io.imread(f'../ACE/ACE/concepts/images/{label}/{img_file}.png')
    HEIGHT, WIDTH, _ = IMG.shape
    DPI = 72 # Check this value
    MASK_SIZE = np.min([(np.min([HEIGHT, WIDTH]) // 100)*100, MAX_MASK_SIZE])
    # MAX_MASKED_IMAGES = 2 * int(np.ceil((HEIGHT * WIDTH) / (MASK_SIZE ** 2))) # scale this with mask size
    MAX_MASKED_IMAGES = MAX_MASKED_IMAGES
    # MAX_MASKS_PER_EPOCH = np.min([int(np.ceil((HEIGHT * WIDTH) / (MASK_SIZE ** 2))), MASKS_PER_EPOCH]) # Max number of searches for each epoch
    MAX_MASKS_PER_EPOCH = MASKS_PER_EPOCH # Max number of searches for each epoch

    # Mask colour
    mean_r_channel = np.mean([item[0]/255 for sublist in IMG for item in sublist])
    mean_g_channel = np.mean([item[1]/255 for sublist in IMG for item in sublist])
    mean_b_channel = np.mean([item[2]/255 for sublist in IMG for item in sublist])
    mask_colour = [mean_r_channel, mean_g_channel, mean_b_channel]

    mask_no = 1
    while mask_no <= MAX_MASKED_IMAGES:

        previous_mask_coords = []
        all_mask_coords = []
        print('mask_no:', mask_no)

        for _ in range(MAX_MASKS_PER_EPOCH):

            CURRENT_SEARCHES = 0
            coords = get_mask_coords(CURRENT_SEARCHES, previous_mask_coords, HEIGHT, WIDTH, MASK_SIZE, 10)

            if not coords:
                break

            all_mask_coords.append(coords)

        fig, ax = plt.subplots(figsize=((IMG.shape[1]/DPI),(IMG.shape[0]/DPI)), dpi=DPI)
        ax.imshow(IMG)
        
        for coords in all_mask_coords:

            mask = draw.rectangle((coords[0],coords[1]), extent=(MASK_SIZE,MASK_SIZE))
            ax.plot(mask[1], mask[0], color=mask_colour, lw=1)

        ax.axis('off')
        plt.tight_layout(pad=0)
        os.makedirs(f'./occluded_images/{dir_name}/{img_filename}/mask_dim_{MASK_SIZE}/mask_no_{mask_no}', exist_ok=True)
        plt.savefig(f'./occluded_images/{dir_name}/{img_filename}/mask_dim_{MASK_SIZE}/mask_no_{mask_no}/mask_no_{mask_no}.JPEG')
        plt.clf()
        mask_coords_df = pd.DataFrame({'x_min':[c[1] for c in all_mask_coords], 'y_min': [c[0] for c in all_mask_coords]})
        mask_coords_df.to_csv(f'./occluded_images/{dir_name}/{img_filename}/mask_dim_{MASK_SIZE}/mask_no_{mask_no}/mask_no_{mask_no}.csv', index=False)
        mask_no += 1


if __name__ == '__main__':

    random.seed(42)
    MAX_MASK_SIZE = 100
    MAX_MASKED_IMAGES = 1000
    MASKS_PER_EPOCH = 1
    # mapping_labels_to_dirs = map_labels_to_dirs()

    baseline_prediction_samples = pd.concat([pd.read_csv(f) for f in glob('./baseline_prediction_samples/*')])
    baseline_prediction_samples_filenames = [filename.split('/')[-2] for filename in baseline_prediction_samples['filename'].values]

    existing_occluded_images = [directory.split('/')[-1] for directory in glob('./occluded_images/**/*')]

    images_to_occlude = list(set(baseline_prediction_samples_filenames) - set(existing_occluded_images))

    for image in [image for image in images_to_occlude if 'n01944390' in image]:
        start = time.time()
        print(f'Starting image: {image}')
        main(image, image.split('_')[0], MAX_MASK_SIZE, MAX_MASKED_IMAGES, MASKS_PER_EPOCH)
        print(f'Finsihed image: {image}')
        end = time.time()
        print('Elapsed time (minutes):', (end - start)/60)
