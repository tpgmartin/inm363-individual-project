from glob import glob
import shutil

IMG_FILES = './superpixels/n09229709/**/**/**_superpixels/*'
training_image_dir = glob(IMG_FILES)

for image_file in training_image_dir:

    image_filename = image_file.split('/')[-1]
    shutil.copyfile(image_file, f'./test_images/bubble/{image_filename}')
