from glob import glob
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA, SparsePCA

label = 'jeep'
layer = 'mixed8'
concepts = np.unique([x.split('_')[2] for x in glob(f'./acts/{label}/*_{layer}')]).tolist()

concepts = [x for x in concepts if 'concept21' or 'concept2' or 'concept13' or 'concept9' in x]

for concept in concepts[:-1]:
    image_acts = np.array([np.load(acts).squeeze() for acts in glob(f'./acts/{label}/acts_{label}_{concept}_*_{layer}')])

    image_acts = [x for x in image_acts if isinstance(x[0], np.float32)]

    pca = PCA(n_components=2)
    pca.fit(image_acts)
    pca_c = pca.components_

    image_acts_embedded = np.dot(image_acts,pca_c.T)

    plt.scatter(image_acts_embedded[:,0], image_acts_embedded[:,1], label=f'{concept}')

plt.legend(loc='upper right')
plt.xlabel('Component 1')
plt.ylabel('Component 2')
plt.savefig(f'{label}_{layer}_concepts_plot.png')
plt.clf()
plt.cla()
plt.close()
