from glob import glob
import matplotlib.pyplot as plt
import os
import pandas as pd
import random
import re
from skimage import io

def init_mask(width, height):

    img_mask = []
    for y in range(height):
        img_mask.append([])
        for _ in range(width):
            img_mask[y].append([0,0,0])
    
    return img_mask

def save_image(image_to_save, image_filename, width, height, dpi, img_filename, MASK_SIZE, is_prediction_correct):

    fig, ax = plt.subplots(figsize=((width/dpi),(height/dpi)), dpi=dpi)
    ax.imshow(image_to_save)
    ax.axis('off')
    plt.tight_layout(pad=0)

    os.makedirs(f"./occlusion_heatmaps/{img_filename.split('_')[0]}/{img_filename}/mask_dim_{MASK_SIZE}/{is_prediction_correct}/{image_filename}_{image_to_save}", exist_ok=True)
    plt.savefig(f"./occlusion_heatmaps/{img_filename.split('_')[0]}/{img_filename}/mask_dim_{MASK_SIZE}/{is_prediction_correct}/{image_filename}_{image_to_save}/{image_filename}_{image_to_save}.JPEG")

def main(f):

    df = pd.read_csv(f)
    df_prediction_true, df_prediction_false = df[df['is_prediction_correct'] == True], df[df['is_prediction_correct'] == False]

    img_filename = df['directory'].values[0].split('/')[-1]
    IMG = io.imread(f"../ACE/ImageNet/ILSVRC2012_img_train/{img_filename.split('_')[0]}/{img_filename}/{img_filename}.JPEG")

    HEIGHT, WIDTH, _ = IMG.shape
    DPI = 72
    MASK_SIZE = int(df['filename'].values[0].split('/')[4].split('mask_dim_')[-1])

    for df in [df_prediction_true, df_prediction_false]:

        is_prediction_correct = df['is_prediction_correct'].values[0]
        
        df_filenames = df['filename'].tolist()
        x_coords = [int(re.findall(r'\+', f.split('/')[-2])[-1]) for f in df_filenames]
        y_coords = [int(re.findall(r'\+', f.split('/')[-2])[-2]) for f in df_filenames]

        # Create blank image mask
        img_mask = init_mask(WIDTH, HEIGHT)
        
        # Apply occlusion masks
        max_intensity = 0
        for x_coord, y_coord in zip(x_coords, y_coords):
            for x in range(x_coord, x_coord+MASK_SIZE):
                for y in range(y_coord, y_coord+MASK_SIZE):
                    
                    img_mask[y][x][0] += 1

                    if img_mask[y][x][0] > max_intensity:
                        max_intensity = img_mask[y][x][0]

        img_cropped_to_mask = IMG.copy()
        img_cropped_to_mask_uniform_intensity = IMG.copy()
        for y in range(HEIGHT):
            for x in range(WIDTH):
                    # Find normalised mask on black background
                    img_mask[y][x][0] /= max_intensity

                    # Crop input image and channel intensities to masks
                    img_cropped_to_mask[y][x][0] *= img_mask[y][x][0]
                    img_cropped_to_mask[y][x][1] *= img_mask[y][x][0]
                    img_cropped_to_mask[y][x][2] *= img_mask[y][x][0]

                    # Crop input image and channel intensities to masks
                    img_cropped_to_mask_uniform_intensity[y][x][0] = img_cropped_to_mask_uniform_intensity[y][x][0] if img_mask[y][x][0] > 0 else 0
                    img_cropped_to_mask_uniform_intensity[y][x][1] = img_cropped_to_mask_uniform_intensity[y][x][0] if img_mask[y][x][0] > 0 else 0
                    img_cropped_to_mask_uniform_intensity[y][x][2] = img_cropped_to_mask_uniform_intensity[y][x][0] if img_mask[y][x][0] > 0 else 0

                    # Overlay masks on original image
                    IMG[y][x][0] = (1-img_mask[y][x][0])*IMG[y][x][0] + 255*img_mask[y][x][0]
                    IMG[y][x][1] = (1-img_mask[y][x][0])*IMG[y][x][1]
                    IMG[y][x][2] = (1-img_mask[y][x][0])*IMG[y][x][2]

        save_image(img_mask, 'mask_heatmap', WIDTH, HEIGHT, DPI, img_filename, MASK_SIZE, is_prediction_correct)
        save_image(img_cropped_to_mask, 'image_cropped_to_mask', WIDTH, HEIGHT, DPI, img_filename, MASK_SIZE, is_prediction_correct)
        save_image(img_cropped_to_mask_uniform_intensity, 'image_cropped_to_mask_uniform_intensity', WIDTH, HEIGHT, DPI, img_filename, MASK_SIZE, is_prediction_correct)
        save_image(IMG, 'image_with_mask', WIDTH, HEIGHT, DPI, img_filename, MASK_SIZE, is_prediction_correct)

if __name__ == '__main__':

    for f in [f for f in glob('occluded_image_predictions/mask_dim_100/*')]:
        main(f'./{f}')
