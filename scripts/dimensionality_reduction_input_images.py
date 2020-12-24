from glob import glob
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA, SparsePCA

# TODO:
# 
# * For bottleneck activations of following image pairs,
# * Bookshop and restaurant, cinema and restaurant
# * Cab and jeep, ambulance and jeep
# * Ant and mantis, damselfly and mantis
# * basketball and balloon
# * Lipstick and lotion
# * Volleyball and basketball
# 
# Find 2D projection of all vectors and plot to grid using image labels

# img_1 = np.load('./acts/ant/acts_ant_n02219486_20946_mixed4c')
# print(img_1)

ants_acts = np.array([np.load(acts).squeeze() for acts in glob('./acts/ant/*')])
mantis_acts = np.array([np.load(acts).squeeze() for acts in glob('./acts/mantis/*')])
basketball_acts = np.array([np.load(acts).squeeze() for acts in glob('./acts/basketball/*')])

pca = PCA(n_components=2)
sparse_pca = SparsePCA(n_components=2, random_state=0)

ants_acts_reduced = pca.fit_transform(ants_acts)
mantis_acts_reduced = pca.fit_transform(mantis_acts)
basketball_acts_reduced = pca.fit_transform(basketball_acts)

ants_acts_sparse_reduced = sparse_pca.fit_transform(ants_acts)
mantis_acts_sparse_reduced = sparse_pca.fit_transform(mantis_acts)
basketball_acts_sparse_reduced = sparse_pca.fit_transform(basketball_acts)

ants_acts_sparse_reduced_x =[c[0] for c in ants_acts_sparse_reduced]
ants_acts_sparse_reduced_y =[c[1] for c in ants_acts_sparse_reduced]

mantis_acts_sparse_reduced_x =[c[0] for c in mantis_acts_sparse_reduced]
mantis_acts_sparse_reduced_y =[c[1] for c in mantis_acts_sparse_reduced]

basketball_acts_sparse_reduced_x =[c[0] for c in basketball_acts_sparse_reduced]
basketball_acts_sparse_reduced_y =[c[1] for c in basketball_acts_sparse_reduced]

# plt.scatter(ants_acts_sparse_reduced_x, ants_acts_sparse_reduced_y)
# plt.scatter(mantis_acts_sparse_reduced_x, mantis_acts_sparse_reduced_y)
# plt.savefig('ants_mantis_acts.png')

plt.scatter(ants_acts_sparse_reduced_x, ants_acts_sparse_reduced_y)
plt.scatter(basketball_acts_sparse_reduced_x, basketball_acts_sparse_reduced_y)
plt.savefig('ants_basketball_acts.png')
