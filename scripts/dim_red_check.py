# TODO: Bring in full funcitonality of `dimensionality_reduction_concepts.py`
from glob import glob
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA, SparsePCA

# chart_type = 'top_5'
# chart_type = 'top_10'
chart_type = 'all_concepts'
label = 'jeep'
layer = 'mixed8'
concepts = np.unique([x.split('_')[2] for x in glob(f'./acts/{label}/*_{layer}') if 'concept' in x]).tolist()

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
    image_acts = np.array([np.load(acts).squeeze() for acts in glob(f'./acts/{label}/acts_{label}_{concept}_*_{layer}')])
    image_acts = [x for x in image_acts if isinstance(x[0], np.float32)]
    all_image_acts.extend(image_acts)

pca = PCA(n_components=2)
pca.fit(all_image_acts)
pca_c = pca.components_

for concept in concepts:
    image_acts = np.array([np.load(acts).squeeze() for acts in glob(f'./acts/{label}/acts_{label}_{concept}_*_{layer}')])
    image_acts = [x for x in image_acts if isinstance(x[0], np.float32)]
    image_acts_embedded = np.dot(image_acts,pca_c.T)

    plt.scatter(image_acts_embedded[:,0], image_acts_embedded[:,1], label=f'{concept}')

if chart_type != 'all_concepts':
    concept_labels = ['Concept ' + concept.split('concept')[-1] for concept in np.unique(top_concepts)]
    plt.legend(concept_labels, loc='upper right')

plt.xlabel('Component 1')
plt.ylabel('Component 2')
plt.savefig(f'./concept_activation_plots/{label}_{layer}_{chart_type}_plot.png')
plt.clf()
plt.cla()
plt.close()
