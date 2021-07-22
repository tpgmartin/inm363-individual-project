from glob import glob
import pandas as pd

def main(filename):

    true_label, image_file, _ = filename.split('/')[-1].split('_image_')
    image_file = image_file.split('_occluded')[0]

    baseline_predictions = pd.read_csv(f'./baseline_prediction_samples/{true_label}baseline_prediction_samples.csv')
    true_label_prediction_probability = baseline_predictions[baseline_predictions['filename'].str.contains(image_file)]['prediction_probability'].values[0]
    
    occluded_image_predictions = pd.read_csv(filename)
    occluded_image_predictions['true_label_prediction_probability_delta'] = occluded_image_predictions['true_label_predictions'] - true_label_prediction_probability

    occluded_image_predictions.to_csv(filename, index=False)
