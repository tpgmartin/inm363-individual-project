# TODO: Bring in full funcitonality of `dimensionality_reduction_concepts.py`
from glob import glob
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA, SparsePCA

concept = 'concept11'
label = 'lipstick'
layer = 'mixed8'
acts = glob(f'./acts/{label}/*{concept}*{layer}')

concept_image_acts = np.array([np.load(acts).squeeze() for acts in glob(f'./acts/{label}/acts_{label}_{concept}_*_{layer}')])
concept_image_acts = [x for x in concept_image_acts if isinstance(x[0], np.float32)]

random_image_acts = np.array([np.load(acts).squeeze() for acts in glob(f'./acts/random/acts_random_random_*_{layer}')])
random_image_acts = [x for x in random_image_acts if isinstance(x[0], np.float32)]
random_image_acts = random_image_acts[:40]

all_image_acts = concept_image_acts + random_image_acts

pca = PCA(n_components=2)
pca.fit(all_image_acts)
pca_c = pca.components_

concept_image_acts_embedded = np.dot(concept_image_acts,pca_c.T)
plt.scatter(concept_image_acts_embedded[:,0], concept_image_acts_embedded[:,1], label=f'{label.capitalize()} {concept}')

random_image_acts_embedded = np.dot(random_image_acts,pca_c.T)
plt.scatter(random_image_acts_embedded[:,0], random_image_acts_embedded[:,1], label='random')

plt.xlabel('Component 1')
plt.ylabel('Component 2')
plt.legend(loc='upper right')
plt.savefig(f'./{label}_{concept}compare_with_random.png')
plt.clf()
plt.cla()
plt.close()
