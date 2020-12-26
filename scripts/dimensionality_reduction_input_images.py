from glob import glob
import matplotlib.pyplot as plt
import numpy as np
import random
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

# TODO
# Get activations for random images
random.seed(1)

pairs = [
    ['bookshop', 'restaurant'],
    ['cinema', 'restaurant'],
    ['cab', 'jeep'],
    ['ambulance', 'jeep'],
    ['ant', 'mantis'],
    ['damselfly', 'mantis'],
    ['bubble', 'balloon'],
    ['lipstick', 'lotion'],
    ['volleyball', 'basketball']
]

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

for input_images in pairs:

    image_1, image_2 = input_images
    random_sample = random.sample(glob('./acts/**/*'), 50)

    image_1_acts = np.array([np.load(acts).squeeze() for acts in glob(f'./acts/{image_1}/*')])
    image_2_acts = np.array([np.load(acts).squeeze() for acts in glob(f'./acts/{image_2}/*')])
    random_acts = np.array([np.load(acts).squeeze() for acts in random_sample])

    pca = PCA(n_components=2)
    # sparse_pca = SparsePCA(n_components=2, random_state=0)

    image_1_acts_reduced = pca.fit_transform(image_1_acts)
    image_2_acts_reduced = pca.fit_transform(image_2_acts)
    random_acts_reduced = pca.fit_transform(random_acts)

    fig = plt.figure(figsize=(12, 5))
    fig.suptitle(f'Activations of {image_1.capitalize()} vs {image_2.capitalize()}')
    plt.subplot(121)
    plot_scatter(image_1, image_2, image_1_acts_reduced, image_2_acts_reduced)
    plt.subplot(122)
    plot_scatter(image_1, 'random', image_1_acts_reduced, random_acts_reduced)
    plt.savefig(f'./pca_acts/{image_1}_{image_2}_pca_acts.png')
    plt.clf()
    plt.cla()
    plt.close()
