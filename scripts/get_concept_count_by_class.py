from glob import glob
import numpy as np

layer = 'mixed8'
acts = glob(f'./acts/**/**concept**{layer}')

concepts_by_class = {}
for act in acts:

    target_class = act.split('/')[2]

    if target_class not in concepts_by_class:
        concepts_by_class[target_class] = {}

    concept = act.split('/')[-1].split('_')[2]

    if concept not in concepts_by_class[target_class]:
        concepts_by_class[target_class][concept] = 1

    else:
        concepts_by_class[target_class][concept] += 1

all_concept_pixels = glob(f'../ACE/ACE/concepts/{layer}*concept*/*.png')

# '../ACE/ACE/concepts/mixed8_mantis_concept9_patches/016_13.png'
superpixel_concept_by_class = {}
for concept in list(concepts_by_class.keys()):

    class_concept_pixels = [concept_pixel.split('/')[-2].split('_')[-1] for concept_pixel in all_concept_pixels if concept_pixel.split('/')[-2].split('_')[-2] == concept]
    superpixel_concept_by_class[concept] = len(np.unique(class_concept_pixels))

print('Available concepts by super-pixel')
for key, value in superpixel_concept_by_class.items():
    print(key, value)

print()
print('Available concept activations by class')
for target_class in concepts_by_class.keys():
    print(target_class, len(concepts_by_class[target_class].keys()))