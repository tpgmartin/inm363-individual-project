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

def load_cav_direction(label_concept, layer, rnd):

    cav_path = f'../ACE/ACE/cavs/{label_concept}-{rnd}-{layer}-linear-0.01.pkl'
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

    concepts = []
    for label in input_images:
        label_concepts = np.unique([label_concept.split('/')[-1].split('-')[0] for label_concept in glob(f'../ACE/ACE/cavs/{label}_concept*-mixed8-*')]).tolist()
        concepts.extend(label_concepts)
    
    concept_pairs = [(c1, c2) for c1 in concepts for c2 in concepts]
    similarity_dic = {pair: [] for pair in concept_pairs}

    sims = [concepts_similarity(concepts, rnd) for rnd in randoms]

    while sims:
      sim = sims.pop()
      for pair in concept_pairs:
        similarity_dic[pair].append(sim[pair])
    
    image_1 = []
    image_2 = []
    values = []
    for sim_keys, sim_values in similarity_dic.items():
        image_1.extend([sim_keys[0]] * len(sim_values))
        image_2.extend([sim_keys[1]] * len(sim_values))
        values.extend(sim_values)
    
    sims_df = pd.DataFrame({
        'concept_1': image_1,
        'concept_2': image_2,
        'cosine_similarity': values
    })
    
    sims_df.to_csv('./cosine_similarities/concept_cavs/concept_cosine_similarities.csv', index=False)

    sims_summary = sims_df.groupby(['concept_1', 'concept_2'])['cosine_similarity'].agg([np.mean, np.std])
    sims_summary.reset_index(inplace=True)
    sims_summary = sims_summary[sims_summary['mean'] <  1]
    sims_summary.sort_values(by=['mean', 'std'], ascending=False, inplace=True)
    sims_summary.to_csv('./cosine_similarities/concept_cavs/concept_summary.csv', index=False)