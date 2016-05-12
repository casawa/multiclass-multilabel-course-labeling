import numpy as np

def convert_to_matrix(data_list):
    list_of_words = [word for word in x[0] for x in data_list]
    tmp = dict(enumerate(list_of_words))
    pos = dict((y,x) for x,y in tmp.iteritems())
    V = len(set(list_of_words))
    X = zeros((len(data_list),V))
    y = zeros((len(data_list),1))
    for i in range(len(data_list)):
        point = data_list[i]
        y[i] = point[1]
        for word in point[0]:
             X[i,pos[word]] = X[i,pos[word]] + 1
    return (X,y)
