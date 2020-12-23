from glob import glob
import numpy as np
import os
import pandas as pd

from helpers import cosine_similarity
from concept_discovery import ConceptDiscovery

if __name__ == '__main__':

    labels = glob('./acts/**/')

    for i in range()

	for label_acts in glob('./acts/**/'):

		label = label_acts.split('/')[-2]

		imgs = glob(f'./acts/{label}/*')
		labels = []
		imgs_1 = []
		imgs_2 = []
		cos_sims = []
		for i in range(len(imgs)):
			for j in range(i, len(imgs)):

				labels.append(label)

				image_1 = imgs[i]
				image_2 = imgs[j]

				image_1_acts = np.load(image_1).squeeze()
				image_2_acts = np.load(image_2).squeeze()

				imgs_1.append(image_1)
				imgs_2.append(image_2)
				cos_sims.append(cosine_similarity(image_1_acts, image_2_acts))
		
		df = pd.DataFrame({
			'label': labels,
			'image_1': imgs_1,
			'image_2': imgs_2,
			'cosine_sim': cos_sims
		})

		df.sort_values(by='cosine_sim', inplace=True, ascending=False)

		df.to_csv(f'./cosine_similarities/different_labels_only/{label}_cosine_similarities.csv', index=False)
		print(f'Found cosine similarities for {label}')
