import pandas as pd
import shutil

# input
df = pd.read_csv('./baseline_prediction_samples/cabbaseline_prediction_samples.csv')
filenames = df['filename']

# destination
dest = '../ACE/ImageNet/ILSVRC2012_img_train/crop_cab/img_sample/cab'

for filename in filenames:
    shutil.copy(filename, dest)
