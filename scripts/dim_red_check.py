# TODO: Bring in full funcitonality of `dimensionality_reduction_concepts.py`
from glob import glob
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA, SparsePCA

chart_type = 'top_10'
# chart_type = 'top_10'
# chart_type = 'all_concepts'

# label = 'ambulance'
# label = 'jeep'
# label = 'mantis'
label = 'cab'
# label = 'police_van'
# label = 'moving_van'
# label = 'shopping_cart'
# label = 'school_bus'
# label = 'bullet_train'
short_label = label.split('_')[0]
layer = 'mixed8'
concepts = np.unique([x.split('_')[2] for x in glob(f'./acts/{short_label}/*_{layer}') if 'concept' in x]).tolist()

# Read from ACE result file top N concepts by TCAV score
if 'top' in chart_type:
    start = 4
    stop = start + int(chart_type.split('_')[-1])

    with open(f'../ACE/ACE/results_summaries/{layer}_{label}_ace_results.txt') as f:
        lines = f.readlines()

    top_concepts = [line.split(':')[1].split('_')[-1] for line in lines[start:stop]]

    concepts = [concept for concept in concepts if concept in top_concepts]

all_image_acts = []
for concept in concepts:
    image_acts = np.array([np.load(acts).squeeze() for acts in glob(f'./acts/{short_label}/acts_{short_label}_{concept}_*_{layer}')])
    image_acts = [x for x in image_acts if isinstance(x[0], np.float32)]
    all_image_acts.extend(image_acts)

pca = PCA(n_components=2)
# pca = SparsePCA(n_components=2,random_state=1)
pca.fit(all_image_acts)
pca_c = pca.components_

acts_names = []
coord_1 = []
coord_2 = []
for concept in concepts:
    acts_filenames = glob(f'./acts/{short_label}/acts_{short_label}_{concept}_*_{layer}')
    image_acts = np.array([np.load(acts).squeeze() for acts in acts_filenames])
    image_acts = [x for x in image_acts if isinstance(x[0], np.float32)]
    image_acts_embedded = np.dot(image_acts,pca_c.T)

    for act_filename, coords in zip(acts_filenames, image_acts_embedded):
        acts_names.append(act_filename)
        coord_1.append(coords[0])
        coord_2.append(coords[1])

    plt.scatter(image_acts_embedded[:,0], image_acts_embedded[:,1], label=f'{concept}')

df = pd.DataFrame({
    'filename': acts_names,
    'component_1': coord_1,
    'component_2': coord_2
})

df.sort_values(by=['component_1','component_2'],inplace=True)

df.to_csv(f'./concept_activation_plots/{label}_{layer}_{chart_type}_mapping.csv', index=False)

if chart_type != 'all_concepts':
    concept_labels = ['Concept ' + concept.split('concept')[-1] for concept in np.unique(top_concepts)]
    plt.legend(concept_labels, loc=(1.02,0.42))

plt.title(f'PCA Plot {label.capitalize()} {layer.capitalize()} Concepts')
plt.xlabel('Component 1')
plt.ylabel('Component 2')
plt.savefig(f'./concept_activation_plots/{label}_{layer}_{chart_type}_plot.png', bbox_inches='tight')
plt.clf()
plt.cla()
plt.close()
