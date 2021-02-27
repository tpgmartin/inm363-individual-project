import argparse
from glob import glob
import numpy as np
import os
import pandas as pd
from shutil import copyfile
import sys

target_class = 'jeep'
layer = 'mixed8'

# Find activations for given target class
activations = glob(f'./acts/{target_class}/acts_{target_class}_concept*_{layer}')

# Find concepts for given target class 
concepts = glob(f'../ACE/ACE/concepts/{layer}_{target_class}_concept*/**/*.png')

# Convert concept paths to format used for activations
concept_activation_filenames = []
for concept in concepts:
    concept_num = concept.split('/')[4].split('_')[-1]
    img_num = concept.split('/')[-2]
    activation_filename = f'./acts/{target_class}/acts_{target_class}_{concept_num}_{img_num}_{layer}'
    concept_activation_filenames.append(activation_filename)

# Then just convert lists of activations to set, find difference, and return as list
missing_activations = set(concept_activation_filenames) - set(activations)
print(f'Missing activations for {target_class.capitalize()}: {len(list(missing_activations))}')
