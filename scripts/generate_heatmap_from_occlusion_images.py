from glob import glob
import matplotlib.pyplot as plt
import numpy as np
import random
from skimage import draw
from skimage import io

def main():

    random.seed(42)

    IMG = io.imread('../ACE/ImageNet/ILSVRC2012_img_train/n03000134/n03000134_8/n03000134_8.JPEG')
    HEIGHT, WIDTH, _ = IMG.shape
    DPI = 72
    MASK_SIZE = 100
    MAX_MASKS = 10

    x_coords = [random.randint(0,(WIDTH-MASK_SIZE)) for _ in range(MAX_MASKS)]
    y_coords = [random.randint(0,(HEIGHT-MASK_SIZE)) for _ in range(MAX_MASKS)]

    img_mask = []
    for y in range(HEIGHT):
        img_mask.append([])
        for _ in range(WIDTH):
            img_mask[y].append([0,0,0])
    
    for x_coord, y_coord in zip(x_coords, y_coords):
        for x in range(x_coord, x_coord+MASK_SIZE):
            for y in range(y_coord, y_coord+MASK_SIZE):
                img_mask[y][x][0] += 1

    max_ = 0
    for y in range(HEIGHT):
        for x in range(WIDTH):
            if img_mask[y][x][0] > max_:
                max_ = img_mask[y][x][0]

    # Save as mask on black background
    for y in range(HEIGHT):
        for x in range(WIDTH):
                img_mask[y][x][0] /= max_

    fig, ax = plt.subplots(figsize=((WIDTH/DPI),(HEIGHT/DPI)), dpi=DPI)
    ax.imshow(img_mask)
        
    ax.axis('off')
    plt.tight_layout(pad=0)
    plt.savefig(f'./test_heatmap.JPEG')

    # Save image cropped to mask
    img_cropped_to_mask = IMG.copy()
    for y in range(HEIGHT):
        for x in range(WIDTH):
            img_cropped_to_mask[y][x][0] *= img_mask[y][x][0]
            img_cropped_to_mask[y][x][1] *= img_mask[y][x][0]
            img_cropped_to_mask[y][x][2] *= img_mask[y][x][0]

    fig, ax = plt.subplots(figsize=((WIDTH/DPI),(HEIGHT/DPI)), dpi=DPI)
    ax.imshow(img_cropped_to_mask)
    ax.axis('off')
    plt.tight_layout(pad=0)
    plt.savefig(f'./image_cropped_to_mask.JPEG')

    # Save image with mask overlaid
    for y in range(HEIGHT):
        for x in range(WIDTH):
            IMG[y][x][0] = (1-img_mask[y][x][0])*IMG[y][x][0] + 255*img_mask[y][x][0]
            IMG[y][x][1] = (1-img_mask[y][x][0])*IMG[y][x][1]
            IMG[y][x][2] = (1-img_mask[y][x][0])*IMG[y][x][2]
            # IMG[y][x][1] *= 0
            # IMG[y][x][2] *= 0

    fig, ax = plt.subplots(figsize=((WIDTH/DPI),(HEIGHT/DPI)), dpi=DPI)
    ax.imshow(IMG)
    ax.axis('off')
    plt.tight_layout(pad=0)
    plt.savefig(f'./image_with_mask.JPEG')

if __name__ == '__main__':

    main(IMG)