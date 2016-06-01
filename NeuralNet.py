from keras.layers import Input, Dense, Flatten
from keras.models import Model, Sequential
from keras.layers.recurrent import LSTM, GRU, SimpleRNN
import numpy as np
from DataModel import DataModel
from glove_utils import Glove

OUTPUT_DIM = 40
EMBED_DIM = 50
WORD_EMBED = 300
NUM_WAYS = 8
prob_threshold = 0.2
NUM_EPOCHS = 100

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
    vocab = []
    for course in train_data:
        tokens = course[0]
        vocab += tokens
       
    return sorted(list(set(vocab)))

def generate_word_vectors(glove, vocab):
    """
    Gets the word vectors for a given vocabulary.
    """
    word_vecs = {}
    for word in vocab:
        word_vecs[word] = glove.process_word(word)

    return word_vecs

def generate_data_wvs(word_vecs, data, max_len):
    """
    Encodes the data into word vectors
    """
    X = []
    for course in data:
        tokens = course[0]
        x = [word_vecs[token] for token in tokens if token in word_vecs]

        # Padding
        while len(x) < max_len:                     
            x.append(np.zeros((WORD_EMBED, )))
        X.append(x)
    return np.array(X)

def compute_labels(data, ways_to_indices):
    """
    Encodes labels in a binary matrix
    """
    labels = np.zeros((len(data), len(ways_to_indices)))
    for i, course in enumerate(data):
        ways = course[1]
        for way in ways:
            labels[i][ways_to_indices[way]] = 1
        
    return labels

def get_ways_to_indices(data_model):
    """
    Generates a map from WAY to index in a vector
    """
    ways = data_model.get_list_of_ways()
    ways_to_indices = {}
    for i, way in enumerate(ways):
        ways_to_indices[way] = i

    return ways_to_indices

def load_data():
    """
    Loads ExploreCourses descriptions and labels, as well as the word vectors.
    """
    glove = Glove(word_vector_size=WORD_EMBED)
    glove.load_glove()
    data_model = DataModel(train_path='data/unstemmed_training_data.txt', test_path='data/unstemmed_testing_data.txt')
    ways_to_indices = get_ways_to_indices(data_model)

    vocab = get_vocab(data_model.training_data_all_ways)
    max_len = get_max_desc_len(data_model.training_data_all_ways + data_model.testing_data_all_ways)
    word_vecs = generate_word_vectors(glove, vocab)
    training_vecs = generate_data_wvs(word_vecs, data_model.training_data_all_ways, max_len)
    testing_vecs = generate_data_wvs(word_vecs, data_model.testing_data_all_ways, max_len)

    training_labels = compute_labels(data_model.training_data_all_ways, ways_to_indices)
    testing_labels = compute_labels(data_model.testing_data_all_ways, ways_to_indices)

    return training_vecs, training_labels, testing_vecs, testing_labels, max_len

def compute_hamming(predicted, labels):
    """
    Calculates the hamming error. 
    """
    hamming = 0.0
    for i, row in enumerate(labels):
        nonzero_labels = []
        nonzero_preds = []
        for j, val in enumerate(row):
            if val != 0:
                nonzero_labels.append(j)

        for j, val in enumerate(predicted[i]):
            if val > prob_threshold:
                nonzero_preds.append(j)

        hamming += float(len(set(nonzero_preds).intersection(set(nonzero_labels))))/len(set(nonzero_preds).union(set(nonzero_labels)))

    return hamming/len(predicted)
            
#def construct_feed_forward_model():
#    """
#    Constructs a feed forward neural net
#    """
    
   

def construct_sequential_model():
    """
    Constructs the sequential neural model.
    """

    model = Sequential()
    #model.add(LSTM(NUM_WAYS, input_length=max_len, input_dim=WORD_EMBED))
    #model.add(GRU(30, input_length=max_len, input_dim=WORD_EMBED))
    model.add(GRU(OUTPUT_DIM, input_dim=WORD_EMBED))
    model.add(Dense(NUM_WAYS, activation='softmax'))

    #inputs = Input(shape=(max_len, WORD_EMBED))

    # AGH description lengths aren't consistent so can't do this without padding...
    #embed = Embedding(output_dim=EMBED_DIM, input_dim=   , input_length=)(inputs)

    #first_layer = Dense(OUTPUT_DIM, activation='sigmoid')(inputs)
    #predictions = Dense(NUM_WAYS, activation='softmax')(first_layer)
    #second_layer = Dense(OUTPUT_DIM, activation='relu')(first_layer)
    #predictions = Dense(OUTPUT_DIM, activation='softmax')(second_layer)

    #model = Model(input=inputs, output=predictions)
    model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])

    return model


def main():
    
    training_vecs, training_labels, testing_vecs, testing_labels, max_len = load_data()
    print training_vecs
    print testing_labels
    print max_len
   
    model = construct_sequential_model()

#    for training_vec, training_label in zip(training_vecs, training_labels):
#        model.train(training_vec, training_label)
    model.fit(training_vecs, training_labels, nb_epoch=NUM_EPOCHS)
#    loss_and_metrics = model.evaluate(testing_vecs, testing_labels)
#    print loss_and_metrics
    predicted = model.predict_proba(testing_vecs)
    print predicted
    print predicted.shape
    print "Hamming Similarity", compute_hamming(predicted, testing_labels)

if __name__ == '__main__':
    main()

