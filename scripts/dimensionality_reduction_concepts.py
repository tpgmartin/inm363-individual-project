from glob import glob
import matplotlib.pyplot as plt
import numpy as np
import random
from sklearn.decomposition import PCA, SparsePCA

random.seed(1)

def plot_scatter(image_1, image_2, image_1_acts, image_2_acts):

    image_1_x =[c[0] for c in image_1_acts]
    image_1_y =[c[1] for c in image_1_acts]

    image_2_x =[c[0] for c in image_2_acts]
    image_2_y =[c[1] for c in image_2_acts]

    plt.scatter(image_1_x, image_1_y, label=f'{image_1}')
    plt.scatter(image_2_x, image_2_y, label=f'{image_2}')
    plt.legend(loc='upper right')
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')

# Concept activations
# ./acts/jeep/acts_jeep_concept1_*_mixed4c
# ./acts/jeep/acts_jeep_concept2_*_mixed4c

# image_blah_acts = np.array([np.load(acts).squeeze() for acts in glob('./acts/jeep/acts_jeep_n03594945_*_mixed4c')])
image_1_acts = np.array([np.load(acts).squeeze() for acts in glob('./acts/jeep/acts_jeep_concept1_*_mixed4c')])
image_2_acts = np.array([np.load(acts).squeeze() for acts in glob('./acts/jeep/acts_jeep_concept2_*_mixed4c')])
# TODO: Get random activations

image_1_acts = [x for x in image_1_acts if isinstance(x[0], np.float32)]
image_2_acts = [x for x in image_2_acts if isinstance(x[0], np.float32)]

pca = PCA(n_components=2)
#     # sparse_pca = SparsePCA(n_components=2, random_state=0)

image_1_acts_reduced = pca.fit_transform(image_1_acts)
image_2_acts_reduced = pca.fit_transform(image_2_acts)
#     random_acts_reduced = pca.fit_transform(random_acts)

#     fig = plt.figure(figsize=(12, 5))
#     fig.suptitle(f'Activations of {image_1.capitalize()} vs {image_2.capitalize()}')
#     plt.subplot(121)
# plot_scatter(image_1, image_2, image_1_acts_reduced, image_2_acts_reduced)
#     plt.subplot(122)
plot_scatter('concept_1', 'concept_2', image_1_acts_reduced, image_2_acts_reduced)
plt.savefig('jeep_example_concepts_plot.png')
# plt.clf()
# plt.cla()
# plt.close()


# pairs = [
#     ['bookshop', 'restaurant'],
#     ['cinema', 'restaurant'],
#     ['cab', 'jeep'],
#     ['ambulance', 'jeep'],
#     ['ant', 'mantis'],
#     ['damselfly', 'mantis'],
#     ['bubble', 'balloon'],
#     ['lipstick', 'lotion'],
#     ['volleyball', 'basketball']
# ]

# for input_images in pairs:

#     image_1, image_2 = input_images
#     random_sample = random.sample(glob('./acts/**/*'), 50)

#     image_1_acts = np.array([np.load(acts).squeeze() for acts in glob(f'./acts/{image_1}/*')])
#     image_2_acts = np.array([np.load(acts).squeeze() for acts in glob(f'./acts/{image_2}/*')])
#     random_acts = np.array([np.load(acts).squeeze() for acts in random_sample])

#     pca = PCA(n_components=2)
#     # sparse_pca = SparsePCA(n_components=2, random_state=0)

#     image_1_acts_reduced = pca.fit_transform(image_1_acts)
#     image_2_acts_reduced = pca.fit_transform(image_2_acts)
#     random_acts_reduced = pca.fit_transform(random_acts)

#     fig = plt.figure(figsize=(12, 5))
#     fig.suptitle(f'Activations of {image_1.capitalize()} vs {image_2.capitalize()}')
#     plt.subplot(121)
#     plot_scatter(image_1, image_2, image_1_acts_reduced, image_2_acts_reduced)
#     plt.subplot(122)
#     plot_scatter(image_1, 'random', image_1_acts_reduced, random_acts_reduced)
#     plt.savefig(f'./pca_acts/{image_1}_{image_2}_pca_acts.png')
#     plt.clf()
#     plt.cla()
#     plt.close()
