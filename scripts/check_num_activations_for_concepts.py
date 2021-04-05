import argparse
from glob import glob
import sys

def main(args):
    label = args.label
    layer = args.layer
    short_label = label.split('_')[0]

    concepts = glob(f'../ACE/ACE/concepts/{layer}_{label}_concept*/*')
    concepts = [concept for concept in concepts if '.png' in concept]
    acts = glob(f'./acts/{short_label}/*{layer}')

    with open(f'../ACE/ACE/results_summaries/{layer}_{label}_ace_results.txt') as f:
        lines = f.readlines()

    top_concepts = [line.split(':')[1].split('_')[-1] for line in lines[4:14]]

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
        try:
            print(concept)
            if images_for_concepts[concept] != acts_for_concepts[concept]:
                print('Missing acts for', concept)
            print('Total super-pixel images:', images_for_concepts[concept])
            print('Total activation files:', acts_for_concepts[concept])
        except KeyError:
            pass

def parse_arguments(argv):
		parser = argparse.ArgumentParser()
		parser.add_argument('--label', type=str,
			help='The name of the target class to be interpreted', default='ambulance')
		parser.add_argument('--layer', type=str,
				help='Names of the target layers of the network (comma separated)',
												default='mixed8')
		return parser.parse_args(argv)


if __name__ == '__main__':

    args = parse_arguments(sys.argv[1:])
    main(args)