import nltk
import csv
import random
from nltk.corpus import movie_reviews
from nltk.classify.scikitlearn import SklearnClassifier
import pickle

from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC

from nltk.classify import ClassifierI
from statistics import mode

from nltk.tokenize import word_tokenize

short_pos = open("reviews/posClean.txt","r").read()
short_neg = open("reviews/disappointing.txt","r").read()

documents = []

for r in short_pos.split('\n'):
    documents.append( (r, "Good") )

for r in short_neg.split('\n'):
    documents.append( (r, "Bad") )


all_words = []

short_pos_words = word_tokenize(short_pos)
short_neg_words = word_tokenize(short_neg)

for w in short_pos_words:
    all_words.append(w.lower())

for w in short_neg_words:
    all_words.append(w.lower())

all_words = nltk.FreqDist(all_words)

word_features = word_features = [w[0] for w in all_words.most_common(1000)]



def find_features(document):
    words = word_tokenize(document)
    features = {}
    for w in word_features:
        features[w] = (w in words)

    return features



featuresets = [(find_features(rev), category) for (rev, category) in documents]

random.shuffle(featuresets)
print(len(featuresets))

testing_set =  featuresets[700:]     
training_set = featuresets[:700]





classifier = nltk.NaiveBayesClassifier.train(training_set)
classifier.show_most_informative_features(100)
print("Original Naive Bayes Algo accuracy percent:", (nltk.classify.accuracy(classifier, testing_set))*100)


def sentiment(text):
    feats = find_features(text)
    prob=classifier.prob_classify(feats)
    r=prob.max()
    q=round(prob.prob("good"),2)
    
    file=open('BOOKsentiment.txt','a')
    file.write(str(classifier.classify(feats)) + "\n")
    file.write(str(q))
    return classifier.classify(feats)






