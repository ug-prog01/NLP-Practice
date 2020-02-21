import nltk
import random
from nltk.corpus import movie_reviews
from nltk.classify.scikitlearn import SklearnClassifier
import pickle
from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernoulliNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC

#from nltk.classify import ClassifierI
from statistics import mode


def my_accuracy(classifier, gold):
	documents, labels = zip(*gold)
	predictions = classifier.classify_documents(documents)
	correct = [y == y_hat for y, y_hat in zip(labels, predictions)]
	if correct:
		return sum(correct) / len(correct)
	else:
		return 0

class VoteClassifier:
	def __init__(self, *classifiers):
		self._classifiers = classifiers

	def classify_documents(self, documents):
		return [self.classify_many(doc) for doc in documents]

	def classify_many(self, features):
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



documents = [(list(movie_reviews.words(fileid)), category)
				for category in movie_reviews.categories()
				for fileid in movie_reviews.fileids(category)]

random.shuffle(documents)

all_words = []

for w in movie_reviews.words():
	all_words.append(w.lower())

all_words = nltk.FreqDist(all_words)

word_features = list(all_words.keys())[:3500]

def find_features(document):
	words = set(document)
	features = {}
	for w in word_features:
		features[w] = (w in words)

	return features

#print((find_features(movie_reviews.words('neg/cv000_29416.txt'))))

featuresets = [(find_features(rev), category) for (rev, category) in documents]

training_set = featuresets[:1900]
testing_set = featuresets[1900:]

# classifier = nltk.NaiveBayesClassifier.train(training_set)

classifier_f = open("naivebayes.pickle", "rb")
classifier = pickle.load(classifier_f)
classifier_f.close()

print("Naive Bayes Algo acuuracy:", (nltk.classify.accuracy(classifier, testing_set))*100)
classifier.show_most_informative_features(15)

MNB_classifier = SklearnClassifier(MultinomialNB())
MNB_classifier.train(training_set)
print("Multinomial Naive Bayes Algo acuuracy:", (nltk.classify.accuracy(MNB_classifier, testing_set))*100)

# GNB_classifier = SklearnClassifier(GaussianNB())
# GNB_classifier.train(training_set)
# print("Gaussian Naive Bayes Algo acuuracy:", (nltk.classify.accuracy(GNB_classifier, testing_set))*100)

BNB_classifier = SklearnClassifier(BernoulliNB())
BNB_classifier.train(training_set)
print("Bernoulli Naive Bayes Algo acuuracy:", (nltk.classify.accuracy(BNB_classifier, testing_set))*100)


LogisticRegression_classifier = SklearnClassifier(LogisticRegression())
LogisticRegression_classifier.train(training_set)
print("LogisticRegression Naive Bayes Algo acuuracy:", (nltk.classify.accuracy(LogisticRegression_classifier, testing_set))*100)


SGDClassifier_classifier = SklearnClassifier(SGDClassifier())
SGDClassifier_classifier.train(training_set)
print("SGDClassifier Classifier Algo acuuracy:", (nltk.classify.accuracy(SGDClassifier_classifier, testing_set))*100)


SVC_classifier = SklearnClassifier(SVC())
SVC_classifier.train(training_set)
print("SVC Classifier Algo acuuracy:", (nltk.classify.accuracy(SVC_classifier, testing_set))*100)


LinearSVC_classifier = SklearnClassifier(LinearSVC())
LinearSVC_classifier.train(training_set)
print("LinearSVC Classifier Algo acuuracy:", (nltk.classify.accuracy(LinearSVC_classifier, testing_set))*100)


NuSVC_classifier = SklearnClassifier(NuSVC())
NuSVC_classifier.train(training_set)
print("NuSVC Classifier Algo acuuracy:", (nltk.classify.accuracy(NuSVC_classifier, testing_set))*100)

voted_classifier = VoteClassifier(classifier, MNB_classifier, BNB_classifier, LogisticRegression_classifier, SGDClassifier_classifier, NuSVC_classifier, SVC_classifier, LinearSVC_classifier)



print("Voted CLassifier accuracy: ", (my_accuracy(voted_classifier, testing_set))*100)

print("Classification:", voted_classifier.classify(testing_set[0][0]), "Confidence%: ", voted_classifier.confidence(testing_set[0][0]))
print("Classification:", voted_classifier.classify(testing_set[1][0]), "Confidence%: ", voted_classifier.confidence(testing_set[1][0]))
print("Classification:", voted_classifier.classify(testing_set[2][0]), "Confidence%: ", voted_classifier.confidence(testing_set[2][0]))
print("Classification:", voted_classifier.classify(testing_set[3][0]), "Confidence%: ", voted_classifier.confidence(testing_set[3][0]))
print("Classification:", voted_classifier.classify(testing_set[4][0]), "Confidence%: ", voted_classifier.confidence(testing_set[4][0]))