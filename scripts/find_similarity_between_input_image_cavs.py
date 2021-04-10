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

def concepts_similarity(concepts, rnd):

    similarity_dic = {}
    for c1 in concepts:
      cav1 = load_cav_direction(c1, layer, rnd)
      for c2 in concepts:
        if (c1, c2) in similarity_dic.keys():
          continue
        cav2 = load_cav_direction(c2, layer, rnd)
        similarity_dic[(c1, c2)] = cosine_similarity(cav1, cav2)
        similarity_dic[(c2, c1)] = similarity_dic[(c1, c2)]
    return similarity_dic

def load_cav_direction(label, layer, rnd):

    cav_path = f'../ACE/ACE/cavs/{label}-{rnd}-{layer}-linear-0.01.pkl'
    vector = cav.CAV.load_cav(cav_path).cavs[0]
    return np.expand_dims(vector, 0) / np.linalg.norm(vector, ord=2)

if __name__ == '__main__':

    input_images = [
        'ambulance',
        'jeep',
        'cab',
        'police_van',
        'moving_van',
        'shopping_cart',
        'school_bus',
        'bullet_train',
        'snail'
    ]
    layer = 'mixed8'
    randoms = ['random500_{}'.format(i) for i in np.arange(20)]

    image_pairs = [(c1, c2) for c1 in input_images for c2 in input_images]
    similarity_dic = {pair: [] for pair in image_pairs}

    sims = [concepts_similarity(input_images, rnd) for rnd in randoms]

    while sims:
      sim = sims.pop()
      for pair in image_pairs:
        similarity_dic[pair].append(sim[pair])
    
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

    sims_df.to_csv('./cosine_similarities/input_images_cavs/input_image_cosine_similarities.csv', index=False)

    sims_summary = sims_df.groupby(['image_1', 'image_2'])['cosine_similarity'].agg([np.mean, np.std])
    sims_summary.reset_index(inplace=True)
    sims_summary = sims_summary[sims_summary['mean'] != 1]
    sims_summary.sort_values(by=['mean', 'std'], ascending=False, inplace=True)
    sims_summary.to_csv('./cosine_similarities/input_images_cavs/input_image_summary.csv', index=False)