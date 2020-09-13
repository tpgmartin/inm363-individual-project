from glob import glob
import pandas as pd

def main(filename):
    
    df = pd.read_csv(filename)
    df['is_prediction_correct'] = df.apply(lambda row: row['true_label'] == row['predicted_label'], axis=1)

    print(df['is_prediction_correct'].value_counts())

    df.to_csv(filename, index=False)

if __name__ == '__main__':

    for f in [f for f in glob('occluded_image_predictions/**/*') if 'hamster' in f]:
        main(f'./{f}')