# INM363 Individual Project

## Notes

### 02.06.20

Have found that `imagenet_labels.txt` used in ACE project are consistent with labels found elsewhere, so can confidently label the image files correctly.

## Installation

Will require [setting up](https://stackoverflow.com/questions/11443302/compiling-numpy-with-openblas-integration) OpenBLAS configuration for scikit-image package

## Instructions

To generate heatmaps from change in prediction probabilities end-to-end run `scripts/main.py`

### Generate Baseline Accuracies

Run the script `get_benchmark_accuracies.py` to get the predicted label and prediction accuracies for training images. For these images, generate a random sample using `sample_baseline_images`.

### Occlude Images

Run `occlude_images.py`

### Get Predictions for Occluded Images

Run `get_occluded_image_accuracy.py`

Also run one of the following
* `check_true_label_prediction_accuracy.py` - return true label prediction delta
* `check_true_vs_predicted_label_occluded_images.py` - determine whether predicted label is true label

### Generate Heatmap from Occluded Images

Run `generate_heatmap_from_occlusion_images.py` - only need to do this step if have run `check_true_vs_predicted_label_occluded_images.py`

Then run one of,
* `generate_net_heatmap.py`
* `generate_net_heatmap_from_prediction_probabilities.py`

### Check Classification of Heatmaps

Run `check_classification_of_heatmaps.py` to find the predicted label and probability of heatmaps.

### Find Cosine Similarities between Images

In order run,
* `get_image_bottleneck_activations.py`
* `get_image_cosine_similarities.py`
* `get_image_cosine_similarities_different_labels.py`
* Run `dimensionality_reduction_input_images.py` to plot activations of input images using PCA

### Find Concept CAVs for Images

* Find concepts using ACE algorithm `get_concepts_for_selected_images.py` (Run within ACE project)
* For discovered concepts, find activations `get_activations_for_concepts.py`
* `dimensionality_reduction_concepts.py` to plot activations of concepts using PCA