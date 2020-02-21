from nltk.corpus import wordnet

# syns = wordnet.synsets('yield')

# lemmas = syns[0].lemmas()

# synonyms = []
# antonyms = []

# eg = wordnet.synsets("good")
# print(type(eg))
# print(type(eg[0].lemmas()))
# print(eg)
# print(eg[0].lemmas())

# for syn in wordnet.synsets("good"):
# 	for l in syn.lemmas():
# 		synonyms.append(l.name())
# 		if l.antonyms():
# 			antonyms.append(l.antonyms()[0].name())

# print(set(synonyms))
# print(set(antonyms))

# w1 = wordnet.synset('sail.n.01')
# w2 = wordnet.synset('boat.n.01')

# print(w1.wup_similarity(w2))

# w1 = wordnet.synset('cat.n.01')
# w2 = wordnet.synset('lion.n.01')

# print(w1.wup_similarity(w2))

# w1 = wordnet.synset('pen.n.01')
# w2 = wordnet.synset('pencil.n.01')

# print(w1.wup_similarity(w2))