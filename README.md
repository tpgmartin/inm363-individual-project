# INM363 Individual Project

This repo contains the code required for my final year project 'Occlusion Sensitivty & Object Detection for Concept-Based Explanations,' in partical fulfilment of the requirements for the degree of MSc. Data Science at City, University of London.

## Installation

Will require [setting up](https://stackoverflow.com/questions/11443302/compiling-numpy-with-openblas-integration) OpenBLAS configuration for scikit-image package. Other dependencies are specified in the root Pipfile.

## Data

The ImageNet 2012 data is used throughout, corresponding to the original dataset use in the ACE project, which can be downloaded [here](https://academictorrents.com/collection/imagenet-2012).

## Code

Unless specified, all scripts are located in the `./scripts` directory.

### ACE

The `./ace` directory contains the full forked version of the ACE project as used in this project. The forked project in its entirety is available [here](https://github.com/tpgmartin/ACE/tree/test-run). The important differences are the scripts `ace_run_image_class_sample.py`, required to run ACE against an image sample without overwriting previous results, and the files prefixed with `combine_iamges_`, which enable ACE to run using pairs of images from different classes. The script `generate_random_images.py` creates sets of random images required for the ACE scripts to run.

### Yolo

The `./yolo` directory contains the specific modified scripts from the Yolo v5 project used to identify the single largest image element corresponding to an appropriate class label. The full modified forked library is available [here](https://github.com/tpgmartin/yolov5/tree/save-single-crop-image). In order to run, follow the steps in the [original library](https://github.com/ultralytics/yolov5) to set up and ensure ImageNet images are available and follow the directory structure assumed in `run_detect.sh`.

### Generate Baseline Accuracies

Run the script `get_benchmark_accuracies.py` to get the predicted label and prediction accuracies for training images. For these images, generate a random sample using `sample_baseline_images`.

### Occlusion Sensitivity

To generate heatmaps from change in prediction probabilities end-to-end run `scripts/main.py`

#### Occlude Images

Run `occlude_images.py`
- `./occluded_images` contains sets of occluded images by occlusion template size and image class and id
- `./occluded_image_predictions` contains predicted label, and prediction probability of image with and without occlusion
- `./net_occlusion_heatmaps_delta_prob` contains various images with occlusion heatmap overlaid

#### Get Predictions for Occluded Images

Run `get_occluded_image_accuracy.py`

Also run one of the following
* `check_true_label_prediction_accuracy.py` - return true label prediction delta
* `check_true_vs_predicted_label_occluded_images.py` - determine whether predicted label is true label

#### Generate Heat Map & Cropped Images from Occlusion Results

Run `generate_heatmap_from_occlusion_images.py` - only need to do this step if have run `check_true_vs_predicted_label_occluded_images.py`

Then run one of,
* `generate_net_heatmap.py`
* `generate_net_heatmap_from_prediction_probabilities.py`

#### Check Classification of Heatmaps

Run `check_classification_of_heatmaps.py` to find the predicted label and probability of heatmaps.

### Find Cosine Similarities between Images

In order run,
* `get_image_bottleneck_activations.py`
* `get_image_cosine_similarities.py`
* `get_image_cosine_similarities_different_labels.py`

### Find Concept CAVs for Images

* Find concepts using ACE algorithm `get_concepts_for_selected_images.py` (Run within ACE project)
* For discovered concepts, find activations `get_activations_for_concepts.py` - this saves activations to `./acts/<target label>/acts_<target label>_concept<concept number>_<patch number>_<bottleneck layer>`
* `dimensionality_reduction_concepts.py` to plot activations of concepts using PCA

### (WIP) Find Similarity of Images and Concepts CAVs

* scripts/find_similarity_between_input_image_activations.py
* scripts/find_similarity_between_input_image_cavs.py
* scripts/find_similarity_between_concept_activations.py
* scripts/find_similarity_between_concept_cavs.py
* notebooks/Process Concept CAV Cosine Similarities.ipynb
* cosine_similarities/

### Other Useful Scripts

* `check_all_concept_activations_present.py`: Run this to check whether there is are the corresponding activation files available for all concepts found for a given image class
* `get_concept_count_by_class.py`: Check number of concepts with activations available by image class
* `get_activations_for_random_concepts.py`: Get random concepts from images and generate corresponding bottleneck activation
* `plot_cav_accuracies_tcav_scores.py`: Plot charts illustrating CAV accuracies and TCAV scores for discovered concepts using ACE algorithm
* Check [yolov5](https://github.com/tpgmartin/yolov5/tree/save-single-crop-image) fork for object dectection script
    - Updates to `detect.py` to find most likely relevant image subject according to hard-coded rules
    - Update to `run_detect.sh` to run script against relevant image subset

### Visualistions

Run

#### CAV visualisations
* cav_accuracies_histograms
* cav_accuracies_plots

#### Bottleneck activations
* concept_activation_plots: Visualisation of concepts by label
* pca_acts: Comparison of bottleneck activations across labels
* pca_acts_concepts: Comparison of bottleneck activations of concepts across labels

#### TCAV visualisations
* tcav_scores_histograms
* tcav_scores_plots

### Labels

This directory contains the original ImageNet labels, the subset of labels used in ACE, and aa mapping from the image codes to the actual class labels.

### Models

This directory contains the model graph files.
