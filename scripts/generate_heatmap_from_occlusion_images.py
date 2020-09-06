from glob import glob
import matplotlib.pyplot as plt
import random
from skimage import io

def init_mask(width, height):

    img_mask = []
    for y in range(height):
        img_mask.append([])
        for _ in range(width):
            img_mask[y].append([0,0,0])
    
    return img_mask

def save_image(image_to_save, filepath, width, height, dpi):

    fig, ax = plt.subplots(figsize=((width/dpi),(height/dpi)), dpi=dpi)
    ax.imshow(image_to_save)
    ax.axis('off')
    plt.tight_layout(pad=0)
    plt.savefig(filepath)

def main():

    random.seed(42)

    IMG = io.imread('../ACE/ImageNet/ILSVRC2012_img_train/n03000134/n03000134_8/n03000134_8.JPEG')
    HEIGHT, WIDTH, _ = IMG.shape
    DPI = 72
    MASK_SIZE = 100
    MAX_MASKS = 10

    x_coords = [random.randint(0,(WIDTH-MASK_SIZE)) for _ in range(MAX_MASKS)]
    y_coords = [random.randint(0,(HEIGHT-MASK_SIZE)) for _ in range(MAX_MASKS)]

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
    for y in range(HEIGHT):
        for x in range(WIDTH):
                # Find normalised mask on black background
                img_mask[y][x][0] /= max_intensity

                # Crop input image and channel intensities to masks
                img_cropped_to_mask[y][x][0] *= img_mask[y][x][0]
                img_cropped_to_mask[y][x][1] *= img_mask[y][x][0]
                img_cropped_to_mask[y][x][2] *= img_mask[y][x][0]

                # Overlay masks on original image
                IMG[y][x][0] = (1-img_mask[y][x][0])*IMG[y][x][0] + 255*img_mask[y][x][0]
                IMG[y][x][1] = (1-img_mask[y][x][0])*IMG[y][x][1]
                IMG[y][x][2] = (1-img_mask[y][x][0])*IMG[y][x][2]

    save_image(img_mask, f'./test_heatmap.JPEG', WIDTH, HEIGHT, DPI)
    save_image(img_cropped_to_mask, f'./image_cropped_to_mask.JPEG', WIDTH, HEIGHT, DPI)
    save_image(IMG, f'./image_with_mask.JPEG', WIDTH, HEIGHT, DPI)

if __name__ == '__main__':

    main()
    # main(IMG)