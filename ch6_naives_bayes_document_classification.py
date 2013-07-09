"""
Natural Language Processing with Python - Chapter 6

http://nltk.org/book/ch06.html
"""

"""
Categorising movies into positive or negative.
"""

import random
import nltk
from nltk.corpus import movie_reviews

documents = [
    (list(movie_reviews.words(fileid)), category)
    for category in movie_reviews.categories()
    for fileid in movie_reviews.fileids(category)
]
random.shuffle(documents)

"""
Feature extraction

Define a feature for each word, whether or not document
contains the word.

Limit number of features to reduce time and space by using
2000 most frequent words in the whole corpus
"""

all_words = nltk.FreqDist(w.lower() for w in movie_reviews.words())
word_features = all_words.keys()[:2000] 

def document_features(document): 
    document_words = set(document) 
    features = {}
    for word in word_features:
        features['contains(%s)' % word] = (word in document_words)
    return features

print str(document_features(movie_reviews.words('pos/cv957_8737.txt')))[:500]
# {'contains(waste)': False, 'contains(lot)': False, ...}

featuresets = [(document_features(d), c) for (d,c) in documents]
train_set, test_set = featuresets[100:], featuresets[:100]
classifier = nltk.NaiveBayesClassifier.train(train_set)

print nltk.classify.accuracy(classifier, test_set) 
# 0.81
classifier.show_most_informative_features(5) 
# Most Informative Features
# contains(outstanding) = True              pos : neg    =     11.1 : 1.0
#      contains(seagal) = True              neg : pos    =      7.7 : 1.0
# contains(wonderfully) = True              pos : neg    =      6.8 : 1.0
#       contains(damon) = True              pos : neg    =      5.9 : 1.0
#      contains(wasted) = True              neg : pos    =      5.8 : 1.0