### Defines Linear Classifier class
from Classifier import Classifier
from sklearn import linear_model
import numpy as np
import util

class LinearClassifier(Classifier):
    """Represents a linear classifier"""
    def __init__(self, data_model, way):
        super(LinearClassifier, self).__init__(data_model,way)
        self.data_model = data_model
        self.way = way
        self.classifier = None
        self.list_of_words = None
        self.tmp = None

    def train(self):
        """Trains model on data from data_model"""
        wc = self.data_model.query_by_way(self.way,True)
        al = self.data_model.get_training_data()
        way_classes = [tuple(elem[0]) for elem in wc]
        all_classes = [tuple(elem[0]) for elem in al]
        neg = [(description,0) for description in all_classes if description not in way_classes]
        pos = [(description,1) for description in all_classes if description not in way_classes]
        data_list = pos + neg
        list_of_words = [word for x in data_list for word in x[0]]
        list_of_words.append("UNK")
        tmp = dict(enumerate(list_of_words))
        self.list_of_words = list_of_words
        self.tmp = tmp
        X,y = util.convert_to_matrix(data_list,tmp)
        clf = linear_model.SGDClassifier()
        y =  np.asarray(y).ravel()
        clf.fit(X,y)
        self.classifier = clf



    def test(self):
        """Tests model on data from data_model"""
        wc = self.data_model.query_by_way(self.way,False)
        al = self.data_model.get_testing_data()
        way_classes = [tuple(elem[0]) for elem in wc]
        all_classes = [tuple(elem[0]) for elem in al]
        neg = [(description,0) for description in all_classes if description not in way_classes]
        pos = [(description,1) for description in all_classes if description not in way_classes]
        data_list = pos + neg
        new_list = []
        for elem in data_list:
            description = list(elem[0])
            label = elem[1]
            for i in range(len(description)):
                if description[i] not in self.list_of_words:
                    description[i] = "UNK"
            new_list.append((tuple(description),label))
        X,y = util.convert_to_matrix(new_list,self.tmp)
        ypred = self.classifier.predict(X)
        err = 0
        for i in range(len(data_list)):
            yi = float(y[i])
            ypi = float(ypred[i])
            if yi != ypi:
                err += 1
        print ypred
        print y
        return float(err)/float(len(data_list))
