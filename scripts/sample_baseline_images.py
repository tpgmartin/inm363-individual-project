from glob import glob
import pandas as pd

def main(true_label, read_path='./baseline_predictions', write_path='./baseline_prediction_samples', sample_size=50):
    
    benchmarks = pd.read_csv(f'{read_path}/{true_label}_baseline_predictions.csv')

    label = ' '.join(true_label.split('_'))

    if label == 'crane bird':
        benchmarks.loc[:,'predicted_label'] = benchmarks['predicted_label'].apply(lambda l: 'crane bird' if l == 'crane' else l)
    elif label == 'african grey':
        benchmarks.loc[:,'predicted_label'] = benchmarks['predicted_label'].apply(lambda l: 'african grey' if l == 'African grey' else l)
    elif label == 'tank suit':
        benchmarks.loc[:,'predicted_label'] = benchmarks['predicted_label'].apply(lambda l: 'tank suit' if l == 'maillot' else l)
    
    benchmarks = benchmarks[(benchmarks['predicted_label'] == label) &
        (benchmarks['predicted_label'] == benchmarks['true_label'])]

    benchmark_sample = benchmarks.sample(n=sample_size, random_state=1)
    benchmark_sample.to_csv(f'{write_path}/{true_label}baseline_prediction_samples.csv', index=False)

if __name__ == '__main__':

    labels = [label.strip() for label in open('./labels/class_labels_subset.txt')]
    existing_samples = [pred.split('/')[-1].split('_baseline_prediction_samples')[0] for pred in glob('./baseline_prediction_samples/*')]
    labels = list(set(labels) - set(existing_samples))

    for label in labels:
        main(label)