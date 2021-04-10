from collections import defaultdict
from glob import glob
import numpy as np
import pandas as pd
from tcav import cav

def cosine_similarity(a, b):
  assert a.shape == b.shape, 'Two vectors must have the same dimensionality'
  a_norm, b_norm = np.linalg.norm(a), np.linalg.norm(b)
  if a_norm * b_norm == 0:
    return 0.
  cos_sim = np.sum(a * b) / (a_norm * b_norm)
  return cos_sim

def norm_vector(acts):
    return acts / np.linalg.norm(acts, ord=2)

if __name__ == '__main__':

    input_images = [
        'ambulance',
        'jeep',
        'cab',
        'police van',
        'moving van',
        'shopping cart',
        'school bus',
        'snail'
    ]
    layer = 'mixed8'

    acts_dic = {}
    for label in input_images:
        acts_dic[label] = [norm_vector(np.load(act).squeeze()) for act in glob(f'./acts/{label}/acts_{label}_n*_{layer}')]

    similarity_dic = defaultdict(list)
    for label_1 in input_images:
        for label_2 in input_images:
            for act_1, act_2 in zip(acts_dic[label_1], acts_dic[label_2]):
                sim = cosine_similarity(act_1, act_2)
                similarity_dic[(label_1, label_2)].append(sim)

    image_1 = []
    image_2 = []
    values = []
    for sim_keys, sim_values in similarity_dic.items():
        image_1.extend([sim_keys[0]] * len(sim_values))
        image_2.extend([sim_keys[1]] * len(sim_values))
        values.extend(sim_values)

    sims_df = pd.DataFrame({
        'image_1': image_1,
        'image_2': image_2,
        'cosine_similarity': values
    })

    sims_df.to_csv('./cosine_similarities/input_images_acts/input_image_cosine_similarities.csv', index=False)
    
    sims_summary = sims_df.groupby(['image_1', 'image_2'])['cosine_similarity'].agg([np.mean, np.std])
    sims_summary.reset_index(inplace=True)
    sims_summary = sims_summary[sims_summary['mean'] < 1]
    sims_summary.sort_values(by=['mean', 'std'], ascending=False, inplace=True)
    sims_summary.to_csv('./cosine_similarities/input_images_acts/input_image_summary.csv', index=False)