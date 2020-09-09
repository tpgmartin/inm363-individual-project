from glob import glob
import matplotlib.pyplot as plt
import numpy as np
import os
from skimage import io

from helpers import map_images_to_labels

def init_mask(width, height):

    img_mask = []
    for y in range(height):
        img_mask.append([])
        for _ in range(width):
            img_mask[y].append([0,0,0])
    
    return img_mask

def save_image(image_to_save, image_filename_to_save, width, height, dpi, img_filename, MASK_SIZE):

    fig, ax = plt.subplots(figsize=((width/dpi),(height/dpi)), dpi=dpi)
    ax.imshow(image_to_save)
    ax.axis('off')
    plt.tight_layout(pad=0)

    os.makedirs(f"./net_occlusion_heatmaps/{img_filename.split('_')[0]}/{img_filename}/mask_dim_{MASK_SIZE}/{img_filename}_{image_filename_to_save}", exist_ok=True)
    plt.savefig(f"./net_occlusion_heatmaps/{img_filename.split('_')[0]}/{img_filename}/mask_dim_{MASK_SIZE}/{img_filename}_{image_filename_to_save}/{img_filename}_{image_filename_to_save}.JPEG")

def main(f):


    try:
        false_heatmap = io.imread(f"./occlusion_heatmaps/{f.split('_')[0]}/{f}/mask_dim_100/is_prediction_correct_false/{f}_mask_heatmap/{f}_mask_heatmap.JPEG")
        true_heatmap = io.imread(f"./occlusion_heatmaps/{f.split('_')[0]}/{f}/mask_dim_100/is_prediction_correct_true/{f}_mask_heatmap/{f}_mask_heatmap.JPEG")
        IMG = io.imread(f"../ACE/ImageNet/ILSVRC2012_img_train/{f.split('_')[0]}/{f}/{f}.JPEG")
    except FileNotFoundError:
        return

    HEIGHT, WIDTH, _ = IMG.shape
    DPI = 72
    MASK_SIZE = 100

    img_mask = init_mask(WIDTH, HEIGHT)

    all_net_intensities = []
    for y in range(HEIGHT):
        for x in range(WIDTH):
            net_intensity = int(false_heatmap[y][x][0]) - int(true_heatmap[y][x][0])

            all_net_intensities.append(net_intensity)
            img_mask[y][x][0] = net_intensity
    
    mean_net_intensity = np.mean(all_net_intensities)
    std_net_intensity = np.std(all_net_intensities)

    all_standardised_intensities = []
    for y in range(HEIGHT):
        for x in range(WIDTH):
            intensity = img_mask[y][x][0]

            # Perform significance testing here
            standardised_intensity = (intensity - mean_net_intensity)/std_net_intensity
            all_standardised_intensities.append(standardised_intensity)
            
            img_mask[y][x][0] = standardised_intensity
    
    max_standardised_intensity = np.max(all_standardised_intensities)
    min_standardised_intensity = np.min(all_standardised_intensities)

    # lower_bound = 1.96 # 95% confidence interval
    # lower_bound = 1.645 # 90% confidence interval
    lower_bound = 1.282 # 80% confidence interval
    # lower_bound = 1 # 68.2% confidence interval
    img_cropped_to_mask = IMG.copy()
    for y in range(HEIGHT):
        for x in range(WIDTH):
            intensity = img_mask[y][x][0]

            if intensity >= lower_bound:
                intensity = (intensity - lower_bound)/(max_standardised_intensity - lower_bound)
                img_mask[y][x][0] = intensity
            elif intensity <= -lower_bound:
                intensity = (abs(intensity) - lower_bound)/(abs(min_standardised_intensity) - lower_bound)
                img_mask[y][x][0] = 0
                img_mask[y][x][2] = intensity
            else:
                img_mask[y][x][0] = 0

            # Crop input image and channel intensities to masks
            img_cropped_to_mask[y][x][0] *= img_mask[y][x][0]
            img_cropped_to_mask[y][x][1] *= img_mask[y][x][0]
            img_cropped_to_mask[y][x][2] *= img_mask[y][x][0]

            IMG[y][x][0] = (1-img_mask[y][x][0]-img_mask[y][x][2])*IMG[y][x][0] + 255*img_mask[y][x][0]
            IMG[y][x][1] = (1-img_mask[y][x][0]-img_mask[y][x][2])*IMG[y][x][1]
            IMG[y][x][2] = (1-img_mask[y][x][0]-img_mask[y][x][2])*IMG[y][x][2] + 255*img_mask[y][x][2]

    save_image(img_mask, 'net_heatmap', WIDTH, HEIGHT, DPI, f, MASK_SIZE)
    save_image(img_cropped_to_mask, 'image_cropped_to_mask', WIDTH, HEIGHT, DPI, f, MASK_SIZE)
    save_image(IMG, 'image_with_mask', WIDTH, HEIGHT, DPI, f, MASK_SIZE)
    plt.clf()

if __name__ == '__main__':

    occlusion_heatmaps = [f.split('/')[-1] for f in glob('./occlusion_heatmaps/**/*')]
    occlusion_heatmaps = [f for f in occlusion_heatmaps if '_' in f]
    existing_heatmaps = [f.split('/')[-1] for f in glob('./net_occlusion_heatmaps/**/*')]
    heatmaps = list(set(occlusion_heatmaps) - set(existing_heatmaps))

    for dir_name in heatmaps:
        main(dir_name)
