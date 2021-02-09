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
# Concepts to check 8, 13, 16, 18
# concept_pairs = [[8, 13], [8, 16], [8, 18], [13, 16], [13, 18], [16, 18]]
# concept_pairs = [[6, 5]]

# for concept_pair in concept_pairs:

# concepts: 1, 2, 3, 4, 5, 8, 13, 16, 18

# TODO: Find set of unique concetps
label = 'ambulance'
layer = 'mixed8'
concepts = np.unique([x.split('_')[2] for x in glob(f'./acts/{label}/*_{layer}')]).tolist()

for concept in concepts:
    # TODO: Find activations for each concept for given network layer
    image_acts = np.array([np.load(acts).squeeze() for acts in glob(f'./acts/{label}/acts_{label}_{concept}_*_{layer}')])
    # image_1_acts = np.array([np.load(acts).squeeze() for acts in glob(f'./acts/jeep/acts_jeep_concept{concept_1}_*_mixed4c')])
    # image_2_acts = np.array([np.load(acts).squeeze() for acts in glob(f'./acts/jeep/acts_jeep_concept{concept_2}_*_mixed4c')])
    # image_3_acts = np.array([np.load(acts).squeeze() for acts in glob(f'./acts/jeep/acts_jeep_concept{concept_3}_*_mixed4c')])
    # image_5_acts = np.array([np.load(acts).squeeze() for acts in glob(f'./acts/jeep/acts_jeep_concept{concept_5}_*_mixed4c')])
    # image_6_acts = np.array([np.load(acts).squeeze() for acts in glob(f'./acts/jeep/acts_jeep_concept{concept_6}_*_mixed4c')])
    # image_7_acts = np.array([np.load(acts).squeeze() for acts in glob(f'./acts/jeep/acts_jeep_concept{concept_7}_*_mixed4c')])
    # image_8_acts = np.array([np.load(acts).squeeze() for acts in glob(f'./acts/jeep/acts_jeep_concept{concept_8}_*_mixed4c')])
    # image_9_acts = np.array([np.load(acts).squeeze() for acts in glob(f'./acts/jeep/acts_jeep_concept{concept_9}_*_mixed4c')])
    # image_2_acts = np.array([np.load(acts).squeeze() for acts in glob(f'./acts/jeep/acts_jeep_concept{concept_2}_*_mixed4c')])
    # image_2_acts = np.array([np.load(acts).squeeze() for acts in glob(f'../ACE/ACE/acts/acts_random500_0_mixed4c')])

    image_acts = [x for x in image_acts if isinstance(x[0], np.float32)]
    # image_2_acts = [x for x in image_2_acts if isinstance(x[0], np.float32)]
    # image_3_acts = [x for x in image_3_acts if isinstance(x[0], np.float32)]
    # image_5_acts = [x for x in image_5_acts if isinstance(x[0], np.float32)]
    # image_6_acts = [x for x in image_6_acts if isinstance(x[0], np.float32)]
    # image_7_acts = [x for x in image_7_acts if isinstance(x[0], np.float32)]
    # image_8_acts = [x for x in image_8_acts if isinstance(x[0], np.float32)]
    # image_9_acts = [x for x in image_9_acts if isinstance(x[0], np.float32)]

    pca = PCA(n_components=2)
    #     # sparse_pca = SparsePCA(n_components=2, random_state=0)

    image_acts_reduced = pca.fit_transform(image_acts)
    # image_2_acts_reduced = pca.fit_transform(image_2_acts)
    # image_3_acts_reduced = pca.fit_transform(image_3_acts)
    # image_5_acts_reduced = pca.fit_transform(image_5_acts)
    # image_6_acts_reduced = pca.fit_transform(image_6_acts)
    # image_7_acts_reduced = pca.fit_transform(image_7_acts)
    # image_8_acts_reduced = pca.fit_transform(image_8_acts)
    # image_9_acts_reduced = pca.fit_transform(image_9_acts)
    #     random_acts_reduced = pca.fit_transform(random_acts)

    #     fig = plt.figure(figsize=(12, 5))
    #     fig.suptitle(f'Activations of {image_1.capitalize()} vs {image_2.capitalize()}')
    #     plt.subplot(121)
    # plot_scatter(image_1, image_2, image_1_acts_reduced, image_2_acts_reduced)
    #     plt.subplot(122)
    # plot_scatter(f'mantis_concept_{concept_1}', f'jeep_concept_{concept_2}', image_1_acts_reduced, image_2_acts_reduced)
    # plot_scatter(f'jeep_concept_{concept_1}', f'random_concept', image_1_acts_reduced, image_2_acts_reduced)
    # plt.savefig(f'mantis_concept_{concept_1}_jeep_concept_{concept_2}_plot.png')
    # plt.savefig(f'jeep_concept_{concept_1}_random_concept_plot.png')

    image_x =[c[0] for c in image_acts_reduced]
    image_y =[c[1] for c in image_acts_reduced]
    plt.scatter(image_x, image_y, label=f'{concept}')

    # image_2_x =[c[0] for c in image_2_acts_reduced]
    # image_2_y =[c[1] for c in image_2_acts_reduced]
    # plt.scatter(image_2_x, image_2_y, label=f'concept_{concept_2}')

    # image_3_x =[c[0] for c in image_3_acts_reduced]
    # image_3_y =[c[1] for c in image_3_acts_reduced]
    # plt.scatter(image_3_x, image_3_y, label=f'concept_{concept_3}')

    # image_5_x =[c[0] for c in image_5_acts_reduced]
    # image_5_y =[c[1] for c in image_5_acts_reduced]
    # plt.scatter(image_5_x, image_5_y, label=f'concept_{concept_5}')

    # image_6_x =[c[0] for c in image_6_acts_reduced]
    # image_6_y =[c[1] for c in image_6_acts_reduced]
    # plt.scatter(image_6_x, image_6_y, label=f'concept_{concept_6}')

    # image_7_x =[c[0] for c in image_7_acts_reduced]
    # image_7_y =[c[1] for c in image_7_acts_reduced]
    # plt.scatter(image_7_x, image_7_y, label=f'concept_{concept_7}')

    # image_8_x =[c[0] for c in image_8_acts_reduced]
    # image_8_y =[c[1] for c in image_8_acts_reduced]
    # plt.scatter(image_8_x, image_8_y, label=f'concept_{concept_8}')

    # image_9_x =[c[0] for c in image_9_acts_reduced]
    # image_9_y =[c[1] for c in image_9_acts_reduced]
    # plt.scatter(image_9_x, image_9_y, label=f'concept_{concept_9}')

plt.legend(loc='upper right')
plt.xlabel('Component 1')
plt.ylabel('Component 2')
plt.savefig(f'{label}_{layer}_concepts_plot.png')
plt.clf()
plt.cla()
plt.close()


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
