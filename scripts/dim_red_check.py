# TODO: Bring in full funcitonality of `dimensionality_reduction_concepts.py`
from glob import glob
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA, SparsePCA

chart_type = 'all_concepts'
label = 'ambulance'
layer = 'mixed8'
concepts = np.unique([x.split('_')[2] for x in glob(f'./acts/{label}/*_{layer}') if 'concept' in x]).tolist()

for concept in concepts:
    image_acts = np.array([np.load(acts).squeeze() for acts in glob(f'./acts/{label}/acts_{label}_{concept}_*_{layer}')])
    image_acts = [x for x in image_acts if isinstance(x[0], np.float32)]

    if len(image_acts) >= 2:
        pca = PCA(n_components=2)
        pca.fit(image_acts)
        pca_c = pca.components_

        image_acts_embedded = np.dot(image_acts,pca_c.T)

        plt.scatter(image_acts_embedded[:,0], image_acts_embedded[:,1], label=f'{concept}')

if chart_type != 'all_concepts':
    plt.legend(loc='upper right')

plt.xlabel('Component 1')
plt.ylabel('Component 2')
plt.savefig(f'./concept_activation_plots/{label}_{layer}_{chart_type}_plot.png')
plt.clf()
plt.cla()
plt.close()
