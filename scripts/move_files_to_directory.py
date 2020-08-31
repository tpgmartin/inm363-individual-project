from glob import glob
import os

IMG_FILES = '../ACE/ImageNet/ILSVRC2012_img_train'
training_image_dirs = glob(f'{IMG_FILES}/*')

total_image_dirs = len(training_image_dirs)
current_image_dir_count = 0

for image_dir in training_image_dirs:

    if current_image_dir_count % 100 == 0:
        print(f'Reorganised {current_image_dir_count} of {total_image_dirs} image directories')

    image_files = glob(f'{image_dir}/*')

    for image_file in image_files:

        image_filename = image_file.split('/')[-1]
        image_file_dir = image_filename.split('.')[0]

        new_image_dir = f'{image_dir}/{image_file_dir}'
        if not os.path.exists(new_image_dir):
            os.mkdir(new_image_dir)
        
        os.rename(image_file, f'{new_image_dir}/{image_filename}')
    
    current_image_dir_count += 1
