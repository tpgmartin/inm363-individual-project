import itertools
import matplotlib.pyplot as plt
import numpy as np
import os
import random
from skimage import draw
from skimage import io
from skimage.util import crop
import time

random.seed(42)

IMG = io.imread('../ACE/ImageNet/ILSVRC2012_img_train/n01531178/n01531178_278.JPEG')
HEIGHT, WIDTH, _ = IMG.shape
DPI = 72

# MAX_MASKS = (IMG.shape[0]*IMG.shape[1])/ (MASK_SIZE**2)
MAX_MASKS = 100
MAX_SEARCHES = 4
MASK_SIZE = 300

execution_times = []

# Mask colour
mean_r_channel = np.mean([item[0]/255 for sublist in IMG for item in sublist])
mean_g_channel = np.mean([item[1]/255 for sublist in IMG for item in sublist])
mean_b_channel = np.mean([item[2]/255 for sublist in IMG for item in sublist])

mask_colour = [mean_r_channel, mean_g_channel, mean_b_channel]

# Make directory for masks of given size
os.mkdir(f'./images/mask_size_{MASK_SIZE}')

def get_mask_coords(CURRENT_SEARCHES):

    global MAX_SEARCHES

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
        if CURRENT_SEARCHES < MAX_SEARCHES:
            return get_mask_coords(CURRENT_SEARCHES)
        else:
            return None
    else:
        previous_mask_coords.extend(mask_coords)
        return (mask_y_min, mask_x_min)

mask_no = 1
while mask_no <= MAX_MASKS:

    previous_mask_coords = []

    for i in range(1,11):

        start = time.time()

        CURRENT_SEARCHES = 0

        coords = get_mask_coords(CURRENT_SEARCHES)

        if not coords:
            break

        mask = draw.rectangle((coords[0],coords[1]), extent=(MASK_SIZE,MASK_SIZE))

        fig, ax = plt.subplots(figsize=((IMG.shape[1]/DPI),(IMG.shape[0]/DPI)), dpi=DPI)
        ax.imshow(IMG)
        ax.plot(mask[1], mask[0], color=mask_colour, lw=1)
        ax.axis('off')
        
        plt.tight_layout(pad=0)
        os.mkdir(f'./images/test_mask_no_{mask_no}_y_min_{coords[0]}_x_min_{coords[1]}')
        plt.savefig(f'./images/test_mask_no_{mask_no}_y_min_{coords[0]}_x_min_{coords[1]}/test_mask_no_{mask_no}_y_min_{coords[0]}_x_min_{coords[1]}.jpg')
        end = time.time()
        execution_times.append(end - start)
    
        mask_no += 1

print(f"Total execution time: {np.sum(execution_times)}")
print(f"Average execution time: {np.mean(execution_times)}")