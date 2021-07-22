#!/bin/bash

FILEPATH=./ACE/ImageNet/ILSVRC2012_img_train/crop_cab/img_sample/cab

# This script shows an example of how to run the code to generate the cropped images
python detect.py --source $FILEPATH --weights yolov5l.pt --conf 0.25 --accepted-names 'truck' 'train' 'car' 'bus' --save-crop