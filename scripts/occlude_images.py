from glob import glob
import itertools
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import random
from skimage import draw
from skimage import io
from skimage.util import crop
import time

from helpers import map_labels_to_dirs

# For a given image find relevant patches with following approach
# • For given image and given patch size, randomly place M patches across N images without duplication
# • Find change in prediction accuracy from baseline prediction for N images
# • For the top X images or more strictly for those images whose prediction changes, overlay the masks
# • Normalise the mask intensities e.g. so max for pixels with largest number of mask overlays
# • Normalised overlay masks determine minimal set of image features required for correct classification

def get_mask_coords(CURRENT_SEARCHES, previous_mask_coords, HEIGHT, WIDTH, MASK_SIZE, MAX_SEARCHES_PER_EPOCH):

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
        if CURRENT_SEARCHES < MAX_SEARCHES_PER_EPOCH:
            return get_mask_coords(CURRENT_SEARCHES, previous_mask_coords, HEIGHT, WIDTH, MASK_SIZE, MAX_SEARCHES_PER_EPOCH)
        else:
            return None
    else:
        previous_mask_coords.extend(mask_coords)
        return (mask_y_min, mask_x_min)

def main(img_filename, dir_name, MAX_MASK_SIZE, SEARCHES_PER_EPOCH):

    IMG = io.imread(f'../ACE/ImageNet/ILSVRC2012_img_train/{dir_name}/{img_filename}/{img_filename}.JPEG')
    HEIGHT, WIDTH, _ = IMG.shape
    DPI = 72 # Check this value
    MASK_SIZE = np.min([(np.min([HEIGHT, WIDTH]) // 100)*100, MAX_MASK_SIZE])
    # MAX_MASKS = 2 * int(np.ceil((HEIGHT * WIDTH) / (MASK_SIZE ** 2))) # scale this with mask size
    MAX_MASKS = 1e3
    MAX_SEARCHES_PER_EPOCH = np.min([int(np.ceil((HEIGHT * WIDTH) / (MASK_SIZE ** 2))), SEARCHES_PER_EPOCH]) # Max number of searches for each epoch

    # Mask colour
    mean_r_channel = np.mean([item[0]/255 for sublist in IMG for item in sublist])
    mean_g_channel = np.mean([item[1]/255 for sublist in IMG for item in sublist])
    mean_b_channel = np.mean([item[2]/255 for sublist in IMG for item in sublist])
    mask_colour = [mean_r_channel, mean_g_channel, mean_b_channel]

    total_count = 1
    mask_no = 1
    previous_mask_coords = []
    while total_count <= MAX_MASKS:

        print('total_count:', total_count)

        for CURRENT_SEARCHES in range(MAX_SEARCHES_PER_EPOCH+1):

            # start = time.time()
            print('CURRENT_SEARCHES:', CURRENT_SEARCHES)
            coords = get_mask_coords(CURRENT_SEARCHES, previous_mask_coords, HEIGHT, WIDTH, MASK_SIZE, MAX_SEARCHES_PER_EPOCH)

            total_count += 1
            if not coords:
                break

            mask = draw.rectangle((coords[0],coords[1]), extent=(MASK_SIZE,MASK_SIZE))

            fig, ax = plt.subplots(figsize=((IMG.shape[1]/DPI),(IMG.shape[0]/DPI)), dpi=DPI)
            ax.imshow(IMG)
            ax.plot(mask[1], mask[0], color=mask_colour, lw=1)
            ax.axis('off')
            
            plt.tight_layout(pad=0)
            os.makedirs(f'./occluded_images/{dir_name}/{img_filename}/mask_dim_{MASK_SIZE}/test_mask_no_{mask_no}_y_min_{coords[0]}_x_min_{coords[1]}', exist_ok=True)
            plt.savefig(f'./occluded_images/{dir_name}/{img_filename}/mask_dim_{MASK_SIZE}/test_mask_no_{mask_no}_y_min_{coords[0]}_x_min_{coords[1]}/test_mask_no_{mask_no}_y_min_{coords[0]}_x_min_{coords[1]}.JPEG')
            # end = time.time()
            # execution_times.append(end - start)
            mask_no += 1


if __name__ == '__main__':

    random.seed(42)
    MAX_MASK_SIZE = 100
    SEARCHES_PER_EPOCH = 10
    # mapping_labels_to_dirs = map_labels_to_dirs()

    baseline_prediction_samples = pd.concat([pd.read_csv(f) for f in glob('./baseline_prediction_samples/*')])
    baseline_prediction_samples_filenames = [filename.split('/')[-2] for filename in baseline_prediction_samples['filename'].values]

    existing_occluded_images = [directory.split('/')[-1] for directory in glob('./occluded_images/**/*')]

    images_to_occlude = list(set(baseline_prediction_samples_filenames) - set(existing_occluded_images))

    for image in images_to_occlude:
        print(f'Starting image: {image}')
        main(image, image.split('_')[0], MAX_MASK_SIZE, SEARCHES_PER_EPOCH)
        print(f'Finsihed image: {image}')
