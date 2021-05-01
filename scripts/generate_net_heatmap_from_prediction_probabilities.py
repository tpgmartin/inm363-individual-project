from glob import glob
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from skimage import io

from helpers import map_images_to_labels

# 1. Read in occlusion prediction probabilities
# 2. Find mean and standard deviation of change in prediction probabilities
# 3. Standardise prediction probabilities
# 4. Choose confidence interval to select only those within certain confidence interval
# 5. Create heatmap from remaining masks

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

    os.makedirs(f"./net_occlusion_heatmaps_delta_prob/{img_filename.split('_')[0]}/{img_filename}/mask_dim_{MASK_SIZE}/{img_filename}_{image_filename_to_save}", exist_ok=True)
    plt.savefig(f"./net_occlusion_heatmaps_delta_prob/{img_filename.split('_')[0]}/{img_filename}/mask_dim_{MASK_SIZE}/{img_filename}_{image_filename_to_save}/{img_filename}_{image_filename_to_save}.JPEG")

def main(f, label, mask_size, z_value):

    IMG = io.imread(f"../ACE/ImageNet/ILSVRC2012_img_train/{f.split('_')[0]}/{f}/{f}.JPEG")

    HEIGHT, WIDTH, _ = IMG.shape
    DPI = 72
    MASK_SIZE = mask_size

    img_mask = init_mask(WIDTH, HEIGHT)

    # bookshop_image_n02871525_10490_occluded_image_predictions
    df = pd.read_csv(f'./occluded_image_predictions/mask_dim_{MASK_SIZE}/{label}_image_{f}_occluded_image_predictions.csv')
    # prediction_probability_delta = df['true_label_prediction_probability_delta'].values

    prob_mean = df['true_label_prediction_probability_delta'].mean()
    prob_std = df['true_label_prediction_probability_delta'].std()
    df['standardised_prediction_prob'] = df['true_label_prediction_probability_delta'].apply(lambda p: (p-prob_mean)/prob_std)

    df = df[(df['standardised_prediction_prob']>z_value)|(df['standardised_prediction_prob']<(-1*z_value))]

    # Negative means does contribute to image prediction so should be red
    # Postiive should be blue

    # Find occlusion masks for remaining rows of df
    # directory,mask_dim,filename,true_label,true_label_predictions,predicted_label,prediction_probability,true_label_prediction_probability_delta

    # for y in range(HEIGHT):
    #     for x in range(WIDTH):
    #         net_intensity = int(false_heatmap[y][x][0]) - int(true_heatmap[y][x][0])

    #         all_net_intensities.append(net_intensity)
    #         img_mask[y][x][0] = net_intensity

    for _, row in df.iterrows():
        filename = row['filename'][:-4] + 'csv'
        if row['standardised_prediction_prob'] < 0:
            pixel_intensity = 1
        elif row['standardised_prediction_prob'] > 0:
            pixel_intensity = -1
            
        mask_coords = pd.read_csv(filename)

        for id_, row in mask_coords.iterrows():
            for y_coord in range(row['y_min'],(row['y_min']+MASK_SIZE)):
                for x_coord in range(row['x_min'],(row['x_min']+MASK_SIZE)):
                    img_mask[y_coord][x_coord][0] += pixel_intensity

                    updated_pixel_intensity = img_mask[row['y_min']][row['x_min']][0]

    pos_vals = []
    neg_vals = []
    for y in range(HEIGHT):
        for x in range(WIDTH):
            pixel_intensity = img_mask[y][x][0]

            if pixel_intensity > 0:
                pos_vals.append(pixel_intensity)
            elif pixel_intensity < 0:
                neg_vals.append(abs(pixel_intensity))

    print('Total pixels:', HEIGHT*WIDTH)
    print('No. positive:', len(pos_vals))
    print('No. negative:', len(neg_vals))
    if len(pos_vals) != 0:
        ub_pos_val = np.max(pos_vals)
        lb_pos_val = np.min(pos_vals)
    else:
        ub_pos_val = 0
        lb_pos_val = 0

    if len(neg_vals) != 0:
        ub_neg_val = np.max(neg_vals)
        lb_neg_val = np.min(neg_vals)
    else:
        ub_neg_val = 0
        lb_neg_val = 0

    img_cropped_to_mask = IMG.copy()
    for y in range(HEIGHT):
        for x in range(WIDTH):
            intensity = img_mask[y][x][0]
            
            if intensity > 0:
                try:
                    intensity = (intensity - lb_pos_val)/(ub_pos_val - lb_pos_val)
                    img_mask[y][x][0] = intensity
                    img_mask[y][x][2] = 0
                except ZeroDivisionError:
                    img_mask[y][x][0] = 0
                    img_mask[y][x][2] = 0
            elif intensity < 0:
                try:
                    intensity = (abs(intensity) - lb_neg_val)/(ub_neg_val - lb_neg_val)
                    img_mask[y][x][0] = 0
                    img_mask[y][x][2] = intensity
                except ZeroDivisionError:
                    img_mask[y][x][0] = 0
                    img_mask[y][x][2] = 0
            else:
                img_mask[y][x][0] = 0
                img_mask[y][x][2] = 0

            # Crop input image and channel intensities to masks
            mean_pixel_intensity = np.mean(IMG)
            if img_mask[y][x][0] == 0:
                img_cropped_to_mask[y][x][0] = mean_pixel_intensity
                img_cropped_to_mask[y][x][1] = mean_pixel_intensity
                img_cropped_to_mask[y][x][2] = mean_pixel_intensity
            else:
                pass

            try:
                IMG[y][x][0] = (1-img_mask[y][x][0]-img_mask[y][x][2])*IMG[y][x][0] + 255*img_mask[y][x][0]
                IMG[y][x][1] = (1-img_mask[y][x][0]-img_mask[y][x][2])*IMG[y][x][1]
                IMG[y][x][2] = (1-img_mask[y][x][0]-img_mask[y][x][2])*IMG[y][x][2] + 255*img_mask[y][x][2]
            except ValueError:
                IMG[y][x][0] = (1-img_mask[y][x][0])*IMG[y][x][0] + 255*img_mask[y][x][0]
                IMG[y][x][1] = (1-img_mask[y][x][0])*IMG[y][x][1]
                IMG[y][x][2] = (1-img_mask[y][x][0])*IMG[y][x][2]

    save_image(img_mask, 'net_heatmap', WIDTH, HEIGHT, DPI, f, MASK_SIZE)
    save_image(img_cropped_to_mask, 'image_cropped_to_mask', WIDTH, HEIGHT, DPI, f, MASK_SIZE)
    save_image(IMG, 'image_with_mask', WIDTH, HEIGHT, DPI, f, MASK_SIZE)
    plt.clf()

if __name__ == '__main__':

    # mapping = map_images_to_labels()

    occlusion_heatmaps = [f.split('/')[-1] for f in glob('./occluded_image_predictions/**/*')]
    occlusion_heatmaps = [f for f in occlusion_heatmaps if '_' in f]
    existing_heatmaps = [f.split('/')[-1] for f in glob('./net_occlusion_heatmaps/**/*')]
    # heatmaps = list(set(occlusion_heatmaps) - set(existing_heatmaps))

    heatmaps = [f for f in occlusion_heatmaps if 'n09229709_28418' in f]
    for f in heatmaps:
        print(f)
        main(('_').join(f.split('_')[2:4]), f.split('_')[0], 1.96)