from glob import glob
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA, SparsePCA

layer = 'mixed8'
label = 'random'
concept = 'random'

image_acts = np.array([np.load(acts).squeeze() for acts in glob(f'./acts/{label}/acts_{label}_{concept}_*_{layer}')])
image_acts = [x for x in image_acts if isinstance(x[0], np.float32)]

pca = PCA(n_components=2)
pca.fit(image_acts)
pca_c = pca.components_

image_acts_embedded = np.dot(image_acts,pca_c.T)

plt.scatter(image_acts_embedded[:,0], image_acts_embedded[:,1], label=f'{concept}')

plt.xlabel('Component 1')
plt.ylabel('Component 2')
plt.savefig(f'./random_{layer}_plot.png')
plt.clf()
plt.cla()
plt.close()
