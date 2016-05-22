from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import accuracy_score
import util
import DataModel as dm
import numpy as np

def construct_labels_matrix(data_model):
    Xt = None
    ymap = {}
    for way in data_model.get_list_of_ways():
        way_classes = set([tuple(x) for x in data_model.query_by_way(way,True)])
        all_classes = set([tuple(x[0]) for x in data_model.get_training_data()])
        tot = [(description,0) for description in all_classes]
        for i in range(len(tot)):
            elem = tot[i]
            desc = elem[0]
            if desc in way_classes:
                tot[i] = (desc,1)
        data_list = tot
        list_of_words = [word for x in data_list for word in x[0]]
        list_of_words.append("UNK")
        list_of_words = list(set(list_of_words))
        tmp = dict(enumerate(list_of_words))
        X,y = util.convert_to_matrix(data_list,tmp)
        Xt = X
        ymap[way] = y
    num_keys = len(ymap.keys())
    num_vals = len(ymap[ymap.keys()[0]])
    y = np.asmatrix(np.zeros((num_keys,num_vals)))
    for i in range(num_keys):
        for j in range(num_vals):
            y[i,j] = ymap[ymap.keys()[i]][j]
    y = y.transpose()
    return (X,y,tmp)

def construct_test_labels_matrix(data_model,temp):
    Xt = None
    ymap = {}
    for way in data_model.get_list_of_ways():
        way_classes = set([tuple(x) for x in data_model.query_by_way(way,False)])
        all_classes = set([tuple(x[0]) for x in data_model.get_testing_data()])
        tot = [(description,0) for description in all_classes]
        for i in range(len(tot)):
            elem = tot[i]
            desc = elem[0]
            if desc in way_classes:
                tot[i] = (desc,1)
        for i in range(len(tot)):
            elem = tot[i]
            desc = elem[0]
            desc = list(desc)
            for j in range(len(desc)):
                if desc[j] not in temp:
                    desc[j] = "UNK"
            desc = tuple(desc)
            tot[i] = (desc,elem[1])
        data_list = tot
        X,y = util.convert_to_matrix(data_list,temp)
        Xt = X
        ymap[way] = y
    num_keys = len(ymap.keys())
    num_vals = len(ymap[ymap.keys()[0]])
    y = np.asmatrix(np.zeros((num_keys,num_vals)))
    for i in range(num_keys):
        for j in range(num_vals):
            y[i,j] = ymap[ymap.keys()[i]][j]
    y = y.transpose()
    return (X,y)

def main():
    data_model = dm.DataModel()
    X,y,tmp = construct_labels_matrix(data_model)
    newX,newy = construct_test_labels_matrix(data_model,tmp)
    clf = OneVsRestClassifier(SVC(kernel='linear'))
    clf.fit(X,y)
    train_pred = clf.predict(X)
    test_pred = clf.predict(newX)
    print "Training error linear: " + str(accuracy_score(train_pred, y))
    print "Testing error linear: " + str(accuracy_score(test_pred, newy))

    X,y,tmp = construct_labels_matrix(data_model)
    newX,newy = construct_test_labels_matrix(data_model,tmp)
    clf2 = OneVsRestClassifier(SVC(kernel='rbf', gamma=10))
    clf2.fit(X,y)
    train_pred = clf2.predict(X)
    test_pred = clf2.predict(newX)
    print "Training error rbf: " + str(accuracy_score(train_pred, y))
    print "Testing error rbf: " + str(accuracy_score(test_pred, newy))


if __name__ == '__main__':
    main()
