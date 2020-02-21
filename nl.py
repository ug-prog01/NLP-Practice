from nltk.tokenize import sent_tokenize, word_tokenize

#tokenizing - word and sentence tokenizers
#corpora - example text

text = "Olo Mr. Shithead, Genki desu ka? You should not eat chalk. Also promote democracy."

print(sent_tokenize(text))
print(word_tokenize(text))


for i in word_tokenize(text):
	print(i)