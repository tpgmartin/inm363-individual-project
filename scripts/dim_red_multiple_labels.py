# TODO: Bring in full funcitonality of `dimensionality_reduction_concepts.py`
from glob import glob
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA, SparsePCA

# def plot_concept_activations(concepts, label, layer, colour):
def plot_concept_activations(concepts, label, layer, colour, embedding):

    all_image_acts = []
    for concept in concepts:

        image_acts = np.array([np.load(acts).squeeze() for acts in glob(f'./acts/{label}/acts_{label}_{concept}_*_{layer}')])
        image_acts = [x for x in image_acts if isinstance(x[0], np.float32)]
        all_image_acts.extend(image_acts)

    image_acts_embedded = np.dot(all_image_acts,embedding.T)

    plt.scatter(image_acts_embedded[:,0], image_acts_embedded[:,1], label=label, c=colour)

# chart_type = 'top_5'
chart_type = 'top_10'
# chart_type = 'all_concepts'
label1 = 'jeep'
label2 = 'ambulance'
layer = 'mixed8'
concepts1 = np.unique([x.split('_')[2] for x in glob(f'./acts/{label1}/*_{layer}') if 'concept' in x]).tolist()
concepts2 = np.unique([x.split('_')[2] for x in glob(f'./acts/{label2}/*_{layer}') if 'concept' in x]).tolist()

# Read from ACE result file top N concepts by TCAV score
all_image_acts = []
if 'top' in chart_type:
    start = 4
    stop = start + int(chart_type.split('_')[-1])

    with open(f'../ACE/ACE/results_summaries/{layer}_{label1}_ace_results.txt') as f:
        lines = f.readlines()
    top_concepts1 = [line.split(':')[1].split('_')[-1] for line in lines[start:stop]]
    concepts1 = [concept for concept in concepts1 if concept in top_concepts1]

    for concept in concepts1:
        image_acts = np.array([np.load(acts).squeeze() for acts in glob(f'./acts/{label1}/acts_{label1}_{concept}_*_{layer}')])
        image_acts = [x for x in image_acts if isinstance(x[0], np.float32)]
        all_image_acts.extend(image_acts)

    with open(f'../ACE/ACE/results_summaries/{layer}_{label2}_ace_results.txt') as f:
        lines = f.readlines()
    top_concepts2 = [line.split(':')[1].split('_')[-1] for line in lines[start:stop]]
    concepts2 = [concept for concept in concepts2 if concept in top_concepts2]

    for concept in concepts2:
        image_acts = np.array([np.load(acts).squeeze() for acts in glob(f'./acts/{label2}/acts_{label2}_{concept}_*_{layer}')])
        image_acts = [x for x in image_acts if isinstance(x[0], np.float32)]
        all_image_acts.extend(image_acts)


pca = PCA(n_components=2)
pca.fit(all_image_acts)
pca_c = pca.components_

plot_concept_activations(concepts1, label1, layer, 'tab:blue', pca_c)
plot_concept_activations(concepts2, label2, layer, 'tab:orange', pca_c)
plt.legend(loc='upper right')

plt.xlabel('Component 1')
plt.ylabel('Component 2')
plt.savefig(f'./concept_activation_plots/{label1}_{label2}_{layer}_{chart_type}_plot.png')
plt.clf()
plt.cla()
plt.close()
