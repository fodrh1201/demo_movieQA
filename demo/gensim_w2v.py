import sys
import numpy as np
import word2vec
import utils
import math
gensim_model = word2vec.load('movie_plots_1364.d-300.mc1.bin')
ignore_word_list = ['.', ',',':', '?', "'s"]
w2v_dim = 300
def encode_w2v_gensim(sentence):
    embedding = np.zeros(300)
    sentence = utils.normalize_alphanumeric(sentence.lower())
    word_list = sentence.split()
    word_size = 0
    for word in word_list:
        if word in ignore_word_list : continue
        try:
            embedding = embedding + gensim_model[word]
            if nan_check(embedding):
                embed()
        except:
            pass

    embedding_norm = np.sum(embedding**2)
    embedding = embedding / (embedding_norm + 1e-6)
    assert embedding.shape == (300, )
    return embedding

def nan_check(arr):
    return math.isnan(np.sum(arr))

