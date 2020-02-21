from nltk.corpus import gutenberg
from nltk.tokenize import sent_tokenize

sample = gutenberg.raw('shakespeare-hamlet.txt')

tij = sent_tokenize(sample)

print(tij)