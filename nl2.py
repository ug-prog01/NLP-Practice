from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

sen = "Some random sentence for NLP."

stop_words = set(stopwords.words("english"))

words = word_tokenize(sen)

filtered = []

filtered = [w for w in words if w not in stop_words]

print(filtered)