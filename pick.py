import nltk
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


class VoteClassifier(ClassifierI):
    def __init__(self, *classifiers):
        self._classifiers = classifiers

    def classify(self, features):
        votes = []
        for c in self._classifiers:
            v = c.classify(features)
            votes.append(v)
        return mode(votes)

    def confidence(self, features):
        votes = []
        for c in self._classifiers:
            v = c.classify(features)
            votes.append(v)

        choice_votes = votes.count(mode(votes))
        conf = choice_votes / len(votes)
        return conf



training_set_f = open("train_data.pickle", "rb")
training_set = pickle.load(training_set_f)
training_set_f.close()

BNB_classifier = SklearnClassifier(BernoulliNB())
BNB_classifier.train(training_set)
#print("Bernoulli Naive Bayes Algo acuuracy:", (nltk.classify.accuracy(BNB_classifier, testing_set))*100)



save_BNB_classifier = open("BNB_classifier.pickle", "wb")
pickle.dump(BNB_classifier, save_BNB_classifier)
save_BNB_classifier.close()
























# short_pos = open("positive.txt", "r").read()
# short_neg = open("negative.txt", "r").read()

# documents = []

# for r in short_pos.split('\n'):
# 	documents.append( (r, "pos") )

# for r in short_neg.split('\n'):
# 	documents.append( (r, "neg") )

# all_words = []


# short_pos_words = word_tokenize(short_pos)
# short_neg_words = word_tokenize(short_neg)

# for w in short_pos_words:
# 	all_words.append(w.lower())

# for w in short_neg_words:
# 	all_words.append(w.lower())

# all_words = nltk.FreqDist(all_words)

# word_features = list(all_words.keys())[:4500]

# def find_features(document):
#     words = set(document)
#     features = {}
#     for w in word_features:
#         features[w] = (w in words)

#     return features

# featuresets = [(find_features(rev), category) for (rev, category) in documents]

# random.shuffle(featuresets)

# training_set = featuresets[:4000]
# testing_set =  featuresets[4000:8000]


# save_test_data = open("test_data.pickle", "wb")
# pickle.dump(testing_set, save_test_data)
# save_test_data.close()

# save_train_data = open("train_data.pickle", "wb")
# pickle.dump(training_set, save_train_data)
# save_train_data.close()




















# print("Naive Bayes Algo acuuracy:", (nltk.classify.accuracy(classifier, testing_set))*100)
# classifier.show_most_informative_features(15)





# LogisticRegression_classifier = SklearnClassifier(LogisticRegression())
# LogisticRegression_classifier.train(training_set)
# print("LogisticRegression Naive Bayes Algo acuuracy:", (nltk.classify.accuracy(LogisticRegression_classifier, testing_set))*100)


# SGDClassifier_classifier = SklearnClassifier(SGDClassifier())
# SGDClassifier_classifier.train(training_set)
# print("SGDClassifier Classifier Algo acuuracy:", (nltk.classify.accuracy(SGDClassifier_classifier, testing_set))*100)


# SVC_classifier = SklearnClassifier(SVC())
# SVC_classifier.train(training_set)
# print("SVC Classifier Algo acuuracy:", (nltk.classify.accuracy(SVC_classifier, testing_set))*100)


# LinearSVC_classifier = SklearnClassifier(LinearSVC())
# LinearSVC_classifier.train(training_set)
# print("LinearSVC Classifier Algo acuuracy:", (nltk.classify.accuracy(LinearSVC_classifier, testing_set))*100)


# NuSVC_classifier = SklearnClassifier(NuSVC())
# NuSVC_classifier.train(training_set)
# print("NuSVC Classifier Algo acuuracy:", (nltk.classify.accuracy(NuSVC_classifier, testing_set))*100)

# save_classifier = open("naivebayes.pickle", "wb")
# pickle.dump(classifier, save_classifier)
# save_classifier.close()











