from glob import glob
import numpy as np
import os
import pandas as pd

from helpers import cosine_similarity
from concept_discovery import ConceptDiscovery

if __name__ == '__main__':

	labels = glob('./acts/**/')

	for i in range(len(labels)-1):
		for j in range(i+1,len(labels)):

			labels_1 = []
			labels_2 = []
			imgs_1 = []
			imgs_2 = []
			cos_sims = []

			if len(glob(f'{labels[i]}*')) != len(glob(f'{labels[j]}*')):
				continue

			for img_1 in glob(f'{labels[i]}*'):
				for img_2 in glob(f'{labels[j]}*'):

					label_1 = labels[i].split('/')[-2]
					label_2 = labels[j].split('/')[-2]

					labels_1.append(label_1)
					labels_2.append(label_2)
					imgs_1.append(img_1)
					imgs_2.append(img_2)

					image_1_acts = np.load(img_1).squeeze()
					image_2_acts = np.load(img_2).squeeze()
					cos_sims.append(cosine_similarity(image_1_acts, image_2_acts))
		
			df = pd.DataFrame({
				'labels_1': labels_1,
				'labels_2': labels_2,
				'image_1': imgs_1,
				'image_2': imgs_2,
				'cosine_sim': cos_sims
			})

			df.sort_values(by='cosine_sim', inplace=True, ascending=False)

			df.to_csv(f"./cosine_similarities/different_labels_only/{labels[i].split('/')[-2]}_{labels[j].split('/')[-2]}_cosine_similarities.csv", index=False)
			print(f"Found cosine similarities for {labels[i].split('/')[-2]} & {labels[j].split('/')[-2]}")
