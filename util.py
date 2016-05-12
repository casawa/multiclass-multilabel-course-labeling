import numpy as np

def convert_to_matrix(data_list):
    list_of_words = [word for x in data_list for word in x[0]]
    tmp = dict(enumerate(list_of_words))
    pos = {}
    for item in tmp:
        pos[tmp[item]] = item
    V = len(set(tmp.keys()))
    X = np.zeros((len(data_list),V))
    y = np.zeros((len(data_list),1))
    print X.shape
    for i in range(len(data_list)):
        point = data_list[i]
        y[i] = point[1]
        for word in point[0]:
             X[i,pos[word]] = X[i,pos[word]] + 1
    return (np.asmatrix(X),np.asmatrix(y))
