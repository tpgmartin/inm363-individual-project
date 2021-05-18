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
        # 'ambulance',
        'jeep',
        # 'cab',
        'police',
        # 'moving',
        # 'shopping',
        # 'school',
        'bullet',
        # 'wine'
        'mantis'
    ]
    layer = 'mixed4c'

    acts_dic = {}
    for label in input_images:
        label_concepts = np.unique(['_'.join(layer_concept.split('/')[-1].split('_')[1:3]) for layer_concept in glob(f'./acts/{label}/acts_{label}_concept*_{layer}')])
        
        for label_concept in label_concepts:
            acts_dic[label_concept] = [norm_vector(np.load(act).squeeze()) for act in glob(f'./acts/{label}/acts_{label_concept}_*_{layer}')]

    similarity_dic = defaultdict(list)
    for label_concept_1 in acts_dic.keys():
        for label_concept_2 in acts_dic.keys():
            for act_1, act_2 in zip(acts_dic[label_concept_1], acts_dic[label_concept_2]):
                sim = cosine_similarity(act_1, act_2)
                similarity_dic[(label_concept_1, label_concept_2)].append(sim)

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

    sims_df.to_csv(f'./cosine_similarities/concept_acts/{layer}_concept_cosine_similarities.csv', index=False)
    
    sims_summary = sims_df.groupby(['concept_1', 'concept_2'])['cosine_similarity'].agg([np.mean, np.std])
    sims_summary.reset_index(inplace=True)
    sims_summary = sims_summary[sims_summary['mean'] < 1]
    sims_summary.sort_values(by=['mean', 'std'], ascending=False, inplace=True)
    sims_summary.to_csv(f'./cosine_similarities/concept_acts/{layer}_concept_summary.csv', index=False)