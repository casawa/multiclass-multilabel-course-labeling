### Defines Linear Classifier class
from Classifier import Classifier
from sklearn import linear_model
import numpy as np
import util
import sys
from sklearn.decomposition import PCA

class LinearPCAClassifier(Classifier):
    """Represents a linear classifier"""
    def __init__(self, data_model, way, n):
        super(LinearPCAClassifier, self).__init__(data_model,way)
        self.data_model = data_model
        self.way = way
        self.classifier = None
        self.list_of_words = None
        self.tmp = None
        self.n = n
        self.pca = PCA(n_components = n)

    def train(self):
        """Trains model on data from data_model"""
        way_classes = set([tuple(x) for x in self.data_model.query_by_way(self.way,True)])
        all_classes = set([tuple(x[0]) for x in self.data_model.get_training_data()])
        neg = [(description,0) for description in all_classes if description not in way_classes]
        pos = [(description,1) for description in all_classes if description in way_classes]
        data_list = pos + neg
        list_of_words = [word for x in data_list for word in x[0]]
        list_of_words.append("UNK")
        list_of_words = list(set(list_of_words))
        tmp = dict(enumerate(list_of_words))
        self.list_of_words = list_of_words
        self.tmp = tmp
        X,y = util.convert_to_PCA_matrix(data_list,tmp,self.pca, True, self.n)
        clf = linear_model.SGDClassifier()
        y =  np.asarray(y).ravel()
        clf.fit(X,y)
        self.classifier = clf
        ypred = self.classifier.predict(X)
        err = 0
        vals = []
        for i in range(len(data_list)):
            yi = float(y[i])
            ypi = float(ypred[i])
            if yi != ypi:
                err += 1
            vals.append((yi,ypi))
        return float(err)/float(len(data_list))


    def test(self):
        """Tests model on data from data_model"""
        way_classes = set([tuple(x) for x in self.data_model.query_by_way(self.way,False)])
        all_classes = set([tuple(x[0]) for x in self.data_model.get_testing_data()])
        neg = [(description,0) for description in all_classes if description not in way_classes]
        pos = [(description,1) for description in all_classes if description in way_classes]
        data_list = pos + neg
        new_list = []
        for elem in data_list:
            description = list(elem[0])
            label = elem[1]
            for i in range(len(description)):
                if description[i] not in self.list_of_words:
                    description[i] = "UNK"
            new_list.append((tuple(description),label))
        X = None
        y = None
        X,y = util.convert_to_PCA_matrix(new_list,self.tmp,self.pca,False,self.n)
        ypred = self.classifier.predict(X)
        err = 0
        for i in range(len(new_list)):
            yi = float(y[i])
            ypi = float(ypred[i])
            if yi != ypi:
                err += 1
        return float(err)/float(len(new_list))

    def classify(self, description):
        for i in range(len(description)):
            if description[i] not in self.list_of_words:
                description[i] = "UNK"
        X,y = util.convert_to_PCA_matrix([(tuple(description),0)],self.tmp,self.pca,False,self.n)
        return int(self.classifier.predict(X))
