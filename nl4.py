import nltk
from nltk.corpus import state_union
from nltk.tokenize import PunktSentenceTokenizer

text = state_union.raw('2006-GWBush.txt')
train_text = state_union.raw('2005-GWBush.txt')

custom_sent_tokenizer = PunktSentenceTokenizer(train_text)

tokenized = custom_sent_tokenizer.tokenize(text)

print(type(custom_sent_tokenizer))
print(tokenized)
print(type(tokenized))

print('\n\n')

def process_content():
	try:
		for i in tokenized:
			words = nltk.word_tokenize(i)
			tagged = nltk.pos_tag(words)

			print(tagged)

	except Exception as e:
		print(str(e))


process_content()
# hello = ['hell', 'stop']
# olo = 'hell man stop it!'
# hell = nltk.word_tokenize(olo)
# print(nltk.pos_tag(hello))