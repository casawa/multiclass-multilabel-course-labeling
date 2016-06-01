from keras.layers import Input, Dense
from keras.models import Model
import numpy as np
from DataModel import DataModel
from glove_utils import Glove

OUTPUT_DIM = 40
EMBED_DIM = 50
WORD_EMBED = 50


def get_max_desc_len(data):
    """
    Returns the maximum length of a course description
    """
    max_len = 0
    for course in data:
        tokens = course[0]
        max_len = max(max_len, len(tokens))

    return max_len

def get_vocab(train_data):
    """
    Generates the vocabulary for the data.
    """
    vocab = set()
    for course in train_data:
        tokens = course[0]
        vocab.union(set(tokens))

    return sorted(list(vocab))

def generate_word_vectors(glove, vocab):
    """
    Gets the word vectors for a given vocabulary.
    """
    word_vecs = {}
    for word in vocab:
        word_vecs[word] = glove.process_word(word)

    return word_vecs

def load_data():
    """
    Loads ExploreCourses descriptions and labels, as well as the word vectors.
    """
    glove = Glove(word_vector_size=WORD_EMBED)
    glove.load_glove()
    data_model = DataModel()
    vocab = get_vocab(data_model.training_data_all_ways)
    max_len = get_max_desc_len(data_model.training_data_all_ways + data_model.testing_data_all_ways)
    word_vecs = generate_word_vectors(glove, vocab)

    # PADDING

def construct_model(max_len):
    """
    Constructs the neural model.
    """
    inputs = Inputs(shape=(max_len, WORD_EMBED))

    # AGH description lengths aren't consistent so can't do this without padding...
    #embed = Embedding(output_dim=EMBED_DIM, input_dim=   , input_length=)(inputs)

    first_layer = Dense(OUTPUT_DIM, activation='sigmoid')(inputs)
    predictions = Dense(OUTPUT_DIM, activation='softmax')(first_layer)
    #second_layer = Dense(OUTPUT_DIM, activation='relu')(first_layer)
    #predictions = Dense(OUTPUT_DIM, activation='softmax')(second_layer)

    model = Model(input=inputs, output=predictions)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    return model

def main():
    
      labels   data  max_len = load_data()
    model = construct_model(max_len)
    model.fit(data, labels)


if __name__ == '__main__':
    main()

