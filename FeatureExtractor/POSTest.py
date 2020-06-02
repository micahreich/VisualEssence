import nltk

tokens = nltk.word_tokenize("I have three dogs. I love my family very much. Computer Science fascinates me.")

POS_tags = nltk.pos_tag(tokens)
_pos = ["NN", "NNS", "NNP", "NNPS", "VB", "VBD", "VBG", "VBP"]
for i in range(len(POS_tags)):
    if POS_tags[i][1] in _pos:
        print(POS_tags[i][0])
