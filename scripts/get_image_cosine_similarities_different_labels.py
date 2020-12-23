from glob import glob
import numpy as np
import os
import pandas as pd

from helpers import cosine_similarity
from concept_discovery import ConceptDiscovery

if __name__ == '__main__':

	# Label pairings to check
	# * Bookshop and restaurant, cinema and restaurant
	# * Cab and jeep, ambulance and jeep
	# * Ant and mantis, damselfly and mantis
	# * Bubble and balloon
	# * Lipstick and lotion
	# * Volleyball and basketball
	label_pairings = [
		['./acts/bookshop/','./acts/restaurant/'],
		['./acts/cinema/','./acts/restaurant/'],
		['./acts/cab/','./acts/jeep/'],
		['./acts/ambulance/','./acts/jeep/'],
		['./acts/ant/','./acts/mantis/'],
		['./acts/damselfly/','./acts/mantis/'],
		['./acts/bubble/','./acts/balloon/'],
		['./acts/lipstick/','./acts/lotion/'],
		['./acts/volleyball/','./acts/basketball/']
	]

	for labels in label_pairings:

		labels_1 = []
		labels_2 = []
		imgs_1 = []
		imgs_2 = []
		cos_sims = []

		if len(glob(f'{labels[0]}*')) != len(glob(f'{labels[1]}*')):
			continue

		for img_1 in glob(f'{labels[0]}*'):
			for img_2 in glob(f'{labels[1]}*'):

				label_1 = labels[0].split('/')[-2]
				label_2 = labels[1].split('/')[-2]

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

		df.to_csv(f"./cosine_similarities/different_labels_only/{labels[0].split('/')[-2]}_{labels[1].split('/')[-2]}_cosine_similarities.csv", index=False)
		print(f"Found cosine similarities for {labels[0].split('/')[-2]} & {labels[1].split('/')[-2]}")
