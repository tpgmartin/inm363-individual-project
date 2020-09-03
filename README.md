# INM363 Individual Project

## Notes

### 02.06.20

Have found that `imagenet_labels.txt` used in ACE project are consistent with labels found elsewhere, so can confidently label the image files correctly.

## Installation

Will require [setting up](https://stackoverflow.com/questions/11443302/compiling-numpy-with-openblas-integration) OpenBLAS configuration for scikit-image package

## Instructions

### Generate Baseline Accuracies

Run the script `get_benchmark_accuracies.py` to get the predicted label and prediction accuracies for training images. For these images, generate a random sample using `sample_baseline_images`.

### Occlude Images

Run `occlude_images.py`
