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

method = 'sparse_pca'
layer = 'mixed8'

pairs = [
    ['ambulance', 'jeep'],
    ['ambulance', 'restaurant'],
    ['ambulance', 'lipstick'],
    ['jeep', 'restaurant'],
    ['jeep', 'lipstick'],
    ['restaurant', 'lipstick']
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

    print(input_images)
    image_1, image_2 = input_images
    all_image_acts = []

    image_1_acts = np.array([np.load(acts).squeeze() for acts in glob(f'./acts/{image_1}/*{layer}')])
    # image_1_acts = [x for x in image_1_acts if isinstance(x[0], np.float32)]
    image_2_acts = np.array([np.load(acts).squeeze() for acts in glob(f'./acts/{image_2}/*{layer}')])
    # image_2_acts = [x for x in image_2_acts if isinstance(x[0], np.float32)]

    all_image_acts.extend(image_1_acts)
    all_image_acts.extend(image_2_acts)

    if method == 'sparse_pca':
        pca = SparsePCA(n_components=2, random_state=0)
    else:
        pca = PCA(n_components=2, random_state=0)

    pca.fit(all_image_acts)
    pca_c = pca.components_
    
    image_1_acts_embedded = np.dot(image_1_acts,pca_c.T)
    image_2_acts_embedded = np.dot(image_2_acts,pca_c.T)

    # fig = plt.figure(figsize=(12, 5))
    plt.title(f'Activations of {image_1.capitalize()} vs {image_2.capitalize()}')
    plot_scatter(image_1, image_2, image_1_acts_embedded, image_2_acts_embedded)
    plt.savefig(f'./pca_acts_concepts/{image_1}_{image_2}_{method}_acts.png')
    plt.clf()
    plt.cla()
    plt.close()
