from glob import glob

label = 'ambulance'
short_label = label.split('_')[0]
layer = 'mixed8'

concepts = glob(f'../ACE/ACE/concepts/{layer}_{label}_concept*/**/*.png')
acts = glob(f'./acts/{short_label}/*{layer}')

# mixed8:restaurant_concept19:0.7025±0.07980444849756184,2.942195744994734e-07
# mixed8:restaurant_concept22:0.67875±0.11132020256898564,9.68045324475897e-07
# mixed8:restaurant_concept7:0.675±0.1414213562373095,0.00036661620810570633
# mixed8:restaurant_concept8:0.6487499999999999±0.13192682630913244,2.158266286688261e-05
# mixed8:restaurant_concept10:0.62625±0.13907619314605935,0.0007136937473238669
with open(f'../ACE/ACE/results_summaries/{layer}_{label}_ace_results.txt') as f:
    lines = f.readlines()

top_concepts = [line.split(':')[1].split('_')[-1] for line in lines[4:14]]

# ../ACE/ACE/concepts/mixed8_restaurant_concept2/016_13/016_13.png
images_for_concepts = {}
for concept_img in concepts:
    concept = concept_img.split('/')[4].split('_')[-1]
    if concept in images_for_concepts:
        images_for_concepts[concept] += 1
    else:
        images_for_concepts[concept] = 1

acts_for_concepts = {}
for act in acts:
    concept = act.split('/')[-1].split('_')[2]
    if concept in acts_for_concepts:
        acts_for_concepts[concept] += 1
    else:
        acts_for_concepts[concept] = 1

for concept in top_concepts:
    print(concept)
    print('Total super-pixel images:', images_for_concepts[concept])
    print('Total activation files:', acts_for_concepts[concept])
    if images_for_concepts[concept] != acts_for_concepts[concept]:
        print('Missing acts for', concept)
