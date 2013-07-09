"""
Natural Language Processing with Python - Chapter 6

http://nltk.org/book/ch06.html
"""

import nltk
import random
from show import show

def gender_features(word):
    return {'last_letter': word[-1]}

names = ([(name, 'male') for name in nltk.corpus.names.words('male.txt')] + \
        [(name, 'female') for name in nltk.corpus.names.words('female.txt')])
random.shuffle(names)
show(names[0:4])

featuresets = [(gender_features(n), g) for (n,g) in names]
train_set, test_set = featuresets[500:], featuresets[:500]

classifier = nltk.NaiveBayesClassifier.train(train_set)

print classifier.classify(gender_features('Neo'))
# 'male'
print classifier.classify(gender_features('Trinity'))
# 'female'
print nltk.classify.accuracy(classifier, test_set)
# 0.758
classifier.show_most_informative_features(5)
# Most Informative Features
#              last_letter = 'a'            female : male   =     38.3 : 1.0
#              last_letter = 'k'              male : female =     31.4 : 1.0
#              last_letter = 'f'              male : female =     15.3 : 1.0
#              last_letter = 'p'              male : female =     10.6 : 1.0
#              last_letter = 'w'              male : female =     10.6 : 1.0

"""
Storing all features uses a large amount of memory for large corpus.
apply_features provides an iterable object that does not store everything
in memory at once.
"""

from nltk.classify import apply_features
train_set = apply_features(gender_features, names[500:])
test_set = apply_features(gender_features, names[:500])

classifier = nltk.NaiveBayesClassifier.train(train_set)

print classifier.classify(gender_features('Neo'))
# 'male'
print classifier.classify(gender_features('Trinity'))
# 'female'
print nltk.classify.accuracy(classifier, test_set)
# 0.758
classifier.show_most_informative_features(5)
# Most Informative Features
#              last_letter = 'a'            female : male   =     38.3 : 1.0
#              last_letter = 'k'              male : female =     31.4 : 1.0
#              last_letter = 'f'              male : female =     15.3 : 1.0
#              last_letter = 'p'              male : female =     10.6 : 1.0
#              last_letter = 'w'              male : female =     10.6 : 1.0

def gender_features2(name):
    features = {}
    features["firstletter"] = name[0].lower()
    features["lastletter"] = name[-1].lower()
    for letter in 'abcdefghijklmnopqrstuvwxyz':
        features["count(%s)" % letter] = name.lower().count(letter)
        features["has(%s)" % letter] = (letter in name.lower())
    return features

print str(gender_features2('John'))[0:100]
# {'count(j)': 1, 'has(d)': False, 'count(b)': 0, ...}

"""
Feature sets returned by gender_features2 contain a large
number of specific features, so it will overfit the small
Names corpus.

Accuracy of the classifier on the test set using gender_features2
is lower than when gender_features was used,
0.748 < 0.758.
"""

random.shuffle(names)
featuresets = [(gender_features2(n), g) for (n,g) in names]
train_set, test_set = featuresets[500:], featuresets[:500]
classifier = nltk.NaiveBayesClassifier.train(train_set)
print nltk.classify.accuracy(classifier, test_set)
# 0.748

"""
Can use error analysis to improve classifier.

Manually look at the mispredictions to understand how they
came about.

More info at https://class.coursera.org/ml-003/lecture/index

Divide up into

- training set
- dev-test / cross validation set
- test set
"""

random.shuffle(names)
train_names = names[1500:]
devtest_names = names[500:1500]
test_names = names[:500]

train_set = [(gender_features(n), g) for (n,g) in train_names]
devtest_set = [(gender_features(n), g) for (n,g) in devtest_names]
test_set = [(gender_features(n), g) for (n,g) in test_names]
classifier = nltk.NaiveBayesClassifier.train(train_set) 
print nltk.classify.accuracy(classifier, devtest_set) 
# 0.765

"""
Generate list of errors.
"""

errors = []
for (name, tag) in devtest_names:
    guess = classifier.classify(gender_features(name))
    if guess != tag:
        errors.append( (tag, guess, name) )

for (tag, guess, name) in sorted(errors)[0:5]: 
    print 'correct=%-8s guess=%-8s name=%-30s' % (tag, guess, name)
# correct=female   guess=male     name=Cindelyn
# correct=female   guess=male     name=Katheryn
# correct=female   guess=male     name=Kathryn
# correct=male     guess=female   name=Aldrich
# correct=male     guess=female   name=Mitch
# correct=male     guess=female   name=Rich

"""
Errors show that:

Some suffixes of more than one gender can indicate name
genders.

eg. -n tends to be male bu -yn tend to be female

Hence error analysis indicates two character features are
important
"""

def gender_features(word):
    return {'suffix1': word[-1:], 'suffix2': word[-2:]}

"""
Testing shows the accuracy is improved by including two
character suffix features.

Use a different dev-test/training split to avoid overfitting.
"""

random.shuffle(names)
train_names = names[1500:]
devtest_names = names[500:1500]
test_names = names[:500]

train_set = [(gender_features(n), g) for (n,g) in train_names]
devtest_set = [(gender_features(n), g) for (n,g) in devtest_names]
classifier = nltk.NaiveBayesClassifier.train(train_set)
print nltk.classify.accuracy(classifier, devtest_set)
# 0.782

