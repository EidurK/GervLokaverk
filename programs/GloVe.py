import numpy as np

file = '../data/GloVe/glove.6B.100d.txt'

class GloVe:
    def __init__(self, glove_file_path=file):
        self.words = {}
        self.embeddings = []
        self.file_path = glove_file_path
        self.load_embeddings()
        self.indx2word = {v: k for k, v in self.words.items()}

    def load_embeddings(self):
        with open(self.file_path, encoding='utf-8') as f:
            i = 0
            for line in f:
                word, coefs = line.split(maxsplit=1)
                coefs = np.fromstring(coefs, "f", sep=" ")
                self.embeddings.append(coefs)
                self.words[word] = i
                i += 1

    def get_vector(self, word):
        if self.words.get(word) is None:
            return None
        return self.embeddings[self.words[word]]
        

    def average_post_vector(self, post):
        word_vectors = np.array([self.get_vector(word) for word in post if word in self.words])
        if word_vectors.size == 0:
            return np.zeros(self.vector_size)
        return np.mean(word_vectors, axis=0)



