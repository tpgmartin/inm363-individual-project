import nltk
from nltk.corpus import wordnet

def get_synonyms(synset):
    for lemma in synset.lemmas():
        for synonym in lemma:
            print(synonym.name())
    # return set([lemma for lemma in synset.lemmas()])

# e.g. dog.n.01 corresponds to the noun "dog"
# synset = wordnet.synsets('cinema')
synset = wordnet.synset('cinema.n.02')
print(synset.lemmas())
# synset = wordnet.synsets('dog')
# print(synset[0].name())
# for words in synset:
#     print(words.name())
# print(synset[0].lemmas()[0].name())
# print(get_synonyms(dog))