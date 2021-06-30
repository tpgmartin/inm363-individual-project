import itertools
import numpy as np
import pandas as pd

def main(target_class, img, MAX_MASK_SIZE, MAX_MASKED_IMAGES, z_val):

    try:
        df = pd.read_csv(f'./occluded_image_predictions/mask_dim_{MAX_MASK_SIZE}/{target_class}_image_{img}_occluded_image_predictions.csv')
        df = df.iloc[:MAX_MASKED_IMAGES]
    except FileNotFoundError:
        return 0

    print(df)
    try: 
        prob_mean = df['true_label_prediction_probability_delta'].mean()
        prob_std = df['true_label_prediction_probability_delta'].std()
        df['standardised_prediction_prob'] = df['true_label_prediction_probability_delta'].apply(lambda p: (p-prob_mean)/prob_std)
    
        return int(df[(df['standardised_prediction_prob']<(-1*z_val))].shape[0] > 0)
    except KeyError:
        return 0


if __name__ == '__main__':

    target_class = 'cab'

    argsLists = [
        [target_class],
        [25, 50, 100],
        [10, 100, 1000],
        [1,1.644854,2.644854]
    ]

    target_classes = []
    mask_sizes = []
    total_masked_images = []
    z_vals = []
    has_occlusion_heatmap = []

    for target_class, MAX_MASK_SIZE, MAX_MASKED_IMAGES, z_val in itertools.product(*argsLists):

        sample = pd.read_csv(f'./baseline_prediction_samples/{target_class}baseline_prediction_samples.csv')
        sampe_imgs = sample['filename']

        for sample_img in sampe_imgs:
    
            img = sample_img.split('/')[-2]
    
            occlusion_heatmap = main(target_class, img, MAX_MASK_SIZE, MAX_MASKED_IMAGES, z_val)

            target_classes.append(target_class)
            mask_sizes.append(MAX_MASK_SIZE)
            total_masked_images.append(MAX_MASKED_IMAGES)
            z_vals.append(z_val)
            has_occlusion_heatmap.append(occlusion_heatmap)

    df = pd.DataFrame({
        'target_class': target_classes,
        'mask_size': mask_sizes,
        'total_masked_images': total_masked_images,
        'z_val': z_vals,
        'has_occlusion_heatmap': has_occlusion_heatmap,
    })

    agg_results = df.groupby(['target_class','mask_size','total_masked_images','z_val'])['has_occlusion_heatmap'].agg([np.sum,'count'])
    agg_results['prop_with_occlusion_results'] = agg_results['sum'] / agg_results['count']
    agg_results.reset_index(inplace=True)

    df.to_csv(f'./occlusion_results/{target_class}_full_occlusion_results.csv', index=False)
    # TODO: Check if only 40 images used in total
    agg_results.to_csv(f'./occlusion_results/{target_class}_summary_occlusion_results.csv', index=False)