from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

ps = PorterStemmer()

words = ['Python', 'Pythoner', 'Pythoned', 'Pythonly', 'Pythoner', 'Pythoning']
'''
for w in words:
	print(ps.stem(w))
'''
sen = "It is very important to be Pythonly while you are Pythoning with Python. All Pythoners have Pythoned various problems."

words = word_tokenize(sen)

for w in words:
	print(ps.stem(w))