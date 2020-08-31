from glob import glob
import itertools
import matplotlib.pyplot as plt
import numpy as np
import os
import random
from skimage import draw
from skimage import io
from skimage.util import crop
import time

from helpers import map_labels_to_dirs

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

def main(label, dir_name, mask_size):
    
    # NB.
    # For `occluded_images` directory structure,
    # `occluded_images` -> `class_code` -> `image_code` -> `occluded_shape_dim` -> `individual_image_code` -> `image_file.jpeg`

    for img_path in glob(f'../ACE/ImageNet/ILSVRC2012_img_train/{dir_name}/*/*.JPEG'):

        img_filename = img_path.split('/')[-2]

        IMG = io.imread(img_path)
        HEIGHT, WIDTH, _ = IMG.shape
        DPI = 72 # Check this value

        MASK_SIZE = mask_size
        MAX_MASKS = int(np.ceil((HEIGHT * WIDTH) / (MASK_SIZE ** 2))) # scale this with mask size
        MAX_SEARCHES_PER_EPOCH = int(np.ceil((HEIGHT * WIDTH) / (MASK_SIZE ** 2))) # Max number of searches for each epoch

        # Mask colour
        mean_r_channel = np.mean([item[0]/255 for sublist in IMG for item in sublist])
        mean_g_channel = np.mean([item[1]/255 for sublist in IMG for item in sublist])
        mean_b_channel = np.mean([item[2]/255 for sublist in IMG for item in sublist])
        mask_colour = [mean_r_channel, mean_g_channel, mean_b_channel]

        mask_no = 1
        while mask_no <= MAX_MASKS:

            previous_mask_coords = []

            for i in range(1,11):

                # start = time.time()

                CURRENT_SEARCHES = 0

                coords = get_mask_coords(CURRENT_SEARCHES, previous_mask_coords, HEIGHT, WIDTH, MASK_SIZE, MAX_SEARCHES_PER_EPOCH)

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

    mapping_labels_to_dirs = map_labels_to_dirs()

    existing_baselines = [' '.join(pred.split('/')[-1].split('_baseline_predictions')[0].split('_')) for pred in glob('./baseline_predictions/*')]

    for label in existing_baselines:
        main(label, mapping_labels_to_dirs[label], 300)
