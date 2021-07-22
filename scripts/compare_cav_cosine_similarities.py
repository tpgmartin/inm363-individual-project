import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

act_sims = pd.read_csv('../cosine_similarities/concept_acts/mixed8_concept_cosine_similarities.csv')

act_sims['label_1'] = act_sims['concept_1'].apply(lambda c: '_'.join(c.split('_')[:-1]))
act_sims['label_2'] = act_sims['concept_2'].apply(lambda c: '_'.join(c.split('_')[:-1]))

print(pd.pivot_table(act_sims[act_sims['label_2'] == 'cab'], values='cosine_similarity', index=['label_1'], columns=['label_2'], aggfunc=[np.mean, np.std]))

print(act_sims[(act_sims['concept_1'] == 'cab_concept10') & (act_sims['label_2'] != 'cab')].sort_values(by=['cosine_similarity'], ascending=False).iloc[:10])

cab_concept = []
ambulance_count = []
bullet_count = []
jeep_count = []
mantis_count = []
police_count = []
for concept in act_sims[act_sims['label_1']=='cab']['concept_1'].unique():

    top_100 = act_sims[(act_sims['concept_1']==concept) & (act_sims['label_2']!='cab')].sort_values(by=['cosine_similarity'], ascending=False)[['label_2','cosine_similarity']].iloc[:100]
    cab_concept.append(concept)
    ambulance_count.append(top_100[top_100['label_2'] == 'ambulance'].shape[0])
    bullet_count.append(top_100[top_100['label_2'] == 'bullet'].shape[0])
    jeep_count.append(top_100[top_100['label_2'] == 'jeep'].shape[0])
    mantis_count.append(top_100[top_100['label_2'] == 'mantis'].shape[0])
    police_count.append(top_100[top_100['label_2'] == 'police'].shape[0])

top_matching_concepts = pd.DataFrame({
    'cab_concept': cab_concept,
    'ambulance_count': ambulance_count,
    'bullet_count': bullet_count,
    'jeep_count': jeep_count,
    'mantis_count': mantis_count,
    'police_count': police_count
})

print(top_matching_concepts)

top_matching_concepts_rank = top_matching_concepts.iloc[:,1:].rank(axis=1, method='min', ascending=False)
print(top_matching_concepts_rank)


print(top_matching_concepts_rank.sum())

print(act_sims[(act_sims['concept_1']=='cab_concept9') & (act_sims['label_2'].str.contains('mantis'))].sort_values(by=['cosine_similarity'], ascending=False))
