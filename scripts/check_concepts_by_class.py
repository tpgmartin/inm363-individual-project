# Determine which concepts have have corresponding activation
from glob import glob
import pandas as pd

layer = 'mixed8'
target_class = 'jeep'

# Get image files for super-pixels
# ../ACE/ACE/concepts/mixed8_jeep_concept1/001_3/001_3.png
# * concept1
# * image number
concept_imgs = [x for x in glob(f'../ACE/ACE/concepts/{layer}_{target_class}_*/**/*.png') if 'patches' not in x]
# print('concept_imgs', len(concept_imgs))
print(len(concept_imgs))

# Activations
# ./acts/jeep/acts_jeep_concept1_001_3_mixed8
activations = glob(f'./acts/{target_class}/acts_{target_class}_concept*_*_{layer}')
activation_concept_imgs = []
for activation in activations:
    l = activation.split('_')
    concept, image_no_1, image_no_2 = l[2], l[3], l[4]
    image_no = f'{image_no_1}_{image_no_2}'
    activation_concept_img = f'../ACE/ACE/concepts/{layer}_{target_class}_{concept}/{image_no}/{image_no}.png'
    activation_concept_imgs.append(activation_concept_img)
print(len(activations))

# ../ACE/ACE/concepts/mixed8_jeep_concept8/018_7/018_7.png

concept_imgs = list(set(concept_imgs) - set(activation_concept_imgs))
print(concept_imgs)