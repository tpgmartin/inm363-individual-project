from collections import defaultdict
import json
import nltk
from nltk.corpus import wordnet
        
def tree():
    return defaultdict(tree)

def add_to_tree(t, path):
    for node in path:
        t = t[node]

label_exceptions = {
    'boxer': 'boxer.n.04',
    'cab': 'cab.n.03',
    'chow': 'chow.n.03',
    'crane_bird': 'crane.n.05',
    'leopard': 'leopard.n.02',
    'liner': 'liner.n.04',
    'punching_bag': 'punching_bag.n.02',
    'reel': 'reel.n.03'
}

# Handle known exceptions
def get_lemmas(label, label_exceptions):
    if label in label_exceptions:
        return label_exceptions[label]
    else:
        return f'{label}.n.01'

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
lemmas = [get_lemmas(label, label_exceptions) for label in labels]
hypernyms = [get_hypernyms(wordnet.synset(lemma), []) for lemma in lemmas]

for hypernym in hypernyms: add_to_tree(wordnet_hierarchy, hypernym)

with open('wordnet_hierarchy.json', 'w+') as f:
    json.dump(wordnet_hierarchy, f)

