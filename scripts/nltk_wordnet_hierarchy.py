import pattern
# from pattern import en
import nltk
from nltk.corpus import wordnet

def get_synonyms(synset):
    for lemma in synset.lemmas():
        for synonym in lemma:
            print(synonym.name())
    # return set([lemma for lemma in synset.lemmas()])

# e.g. dog.n.01 corresponds to the noun "dog"
# synset = wordnet.synsets('cinema')
# synset = wordnet.synset('cinema.n.02')
# print(synset.lemmas())

# Get synonyms
def get_synonyms(synset):
    synonyms = []
    for l in synset.lemmas():
        synonyms.append(l.name())
    return synonyms

# print(set(get_synonyms(wordnet.synset('dog.n.01'))))
print(wordnet.synsets('bookshop'))
print(set(get_synonyms(wordnet.synset('bookshop.n.01'))))

# Get hypernyms
# For bookshop should get
# shop
# v
# place of business, establishment
# v
# establishment
# v
# structure
# v artifact
def get_hypernyms(synset):
    hypernyms = set()
    for hypernym in synset.hypernyms():
        hypernyms |= set(get_hypernyms(hypernym))
    return hypernyms | set(synset.hypernyms())
bookshop = wordnet.synset('bookshop.n.01')
print(get_hypernyms(bookshop))

bookshop = wordnet.synset('bookshop.n.01')
print(bookshop.hypernyms())
# print(bookshop.holonyms())

print(wordnet.synset('bookshop.n.01').holonyms())