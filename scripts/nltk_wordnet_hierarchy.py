from collections import defaultdict
import json
import nltk
from nltk.corpus import wordnet

        
def tree():
    return defaultdict(tree)

def add_to_tree(t, path):

    max_idx = len(path) - 1
    for idx, node in enumerate(path):
        if idx == max_idx:
            t[path[idx-1]] = node
        else:
            t = t[node]

def get_hypernyms(synset, hypernyms=[]):
    synset_hypernyms = synset.hypernyms()
    if len(synset.hypernyms()) == 0:
        hypernyms.reverse()
        return hypernyms
    else:
        if len(hypernyms) == 0:
            hypernyms.append(synset.lemmas()[0].name())
        for synset in synset_hypernyms:
            hypernyms.append(synset_hypernyms[0].lemmas()[0].name())
        return get_hypernyms(synset, hypernyms)

wordnet_hierarchy = tree()

labels = [line.strip() for line in open('./labels/class_labels_subset.txt')]
lemmas = [f'{label}.n.01' if label != 'crane_bird' else 'crane.n.05' for label in labels]
hypernyms = [get_hypernyms(wordnet.synset(lemma), []) for lemma in lemmas]

for hypernym in hypernyms: add_to_tree(wordnet_hierarchy, hypernym)

print(json.dumps(wordnet_hierarchy))