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

			chunkGram = r"""Chunk: {<RB.?>*<VB.?>*<NNP>+<NN>?}
							}<TO|IP>+{"""

			chunkParser = nltk.RegexpParser(chunkGram)
			chunked = chunkParser.parse(tagged)

			chunked.draw()
















	except Exception as e:
		print(str(e))


process_content()