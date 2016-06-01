import os as os
import numpy as np

class Glove:
    
    def __init__(self, word_vector_size=50):
        self.word_vector_size = word_vector_size
        self.word2vec = {}
        self.vocab = {}
        self.ivocab = {}

    def load_glove(self):
        self.word2vec = {}

        print "==> loading glove"
        with open("../../../CS_224D/Project/Dynamic-Memory-Networks/data/glove/glove.6B." + str(self.word_vector_size) + "d.txt") as f:
            for line in f:
                l = line.split()
                self.word2vec[l[0]] = map(float, l[1:])

        print "==> glove is loaded"

        return self.word2vec

    
    def create_vector(self, word, silent=False):
        # if the word is missing from Glove, create some fake vector and store in glove!
        vector = np.random.uniform(0.0,1.0,(self.word_vector_size,))
        self.word2vec[word] = vector
        if (not silent):
            print "utils.py::create_vector => %s is missing" % word
        return vector


    def process_word(self, word, to_return="word2vec", silent=False):
        if not word in self.word2vec:
            self.create_vector(word, silent)
        if not word in self.vocab:
            next_index = len(self.vocab)
            self.vocab[word] = next_index
            self.ivocab[next_index] = word

        if to_return == "word2vec":
            return self.word2vec[word]
        elif to_return == "index":
            return self.vocab[word]
        elif to_return == "onehot":
            raise Exception("to_return = 'onehot' is not implemented yet")

