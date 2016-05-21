#Defines NaiveBayesClassifier Class
from Classifier import Classifier
import numpy as np
import util

class NaiveBayes(Classifier):
    def __init__(self, data_model, way):
        super(NaiveBayes, self).__init__(data_model,way)
        self.phi_y = 0
        self.way = way
        self.phi_x0 = None
        self.phi_x1 = None
        self.word_list = None
        self.V = 0
        self.V = 0


    def train(self):
        way_classes_list = self.data_model.query_by_way(self.way, True)
        all_classes_list = self.data_model.get_training_data()

        way_classes = set([tuple(x) for x in self.data_model.query_by_way(self.way,True)])
        all_classes = set([tuple(x[0]) for x in self.data_model.get_training_data()])
        neg = [(description,0) for description in all_classes if description not in way_classes]
        pos = [(description,1) for description in all_classes if description in way_classes]
        data_list = pos + neg
        X, y = self.convert_to_matrix_naive(data_list)
        m = X.shape[0]
        self.phi_x0 = np.ones((X.shape[1], 1))
        self.phi_x1 = np.ones((X.shape[1], 1))
        for index, row in enumerate(X):
            if y[index] == 0:
                self.phi_x0 += np.transpose(row)
            else:
                self.phi_x1 += np.transpose(row)
                self.phi_y += 1
        self.phi_x1 /= (self.phi_y + 2)
        self.phi_x0 /= (m + 2 - self.phi_y)
        self.phi_y /= float(m)




        self.phi_x1 = np.log(self.phi_x1)
        self.phi_x0 = np.log(self.phi_x0)
        errors = 0
        for index, row in enumerate(X):
            if self.get_predicted_class(row, np.log(self.phi_y), np.log(1 - self.phi_y)) != y[index]:
                errors += 1
        errors /= float(m)
        return errors

    def test(self):
        """Tests model on data from data_model"""
        way_classes = set([tuple(x) for x in self.data_model.query_by_way(self.way,False)])
        all_classes = set([tuple(x[0]) for x in self.data_model.get_testing_data()])
        neg = [(description,0) for description in all_classes if description not in way_classes]
        pos = [(description,1) for description in all_classes if description in way_classes]
        data_list = pos + neg
        X, y = self.convert_to_matrix_test(data_list)
        m = X.shape[0]
        errors = 0
        for index, row in enumerate(X):
            if self.get_predicted_class(row, np.log(self.phi_y), np.log(1 - self.phi_y)) != y[index]:
                errors += 1
        return errors / float(m)

    def classify(self, description):
        X = np.zeros((1, self.V))
        for word in description:
            if word in self.word_list:
                X[0, self.word_list[word]] = 1
            else:
                X[0, self.word_list['NOTAWORD']] = 1
        return self.get_predicted_class(X, np.log(self.phi_y), np.log(1 - self.phi_y))

    def get_predicted_class(self,x, log_y, log_y1):
        total_0 = float(x * self.phi_x0) + log_y
        total_1 = float(x * self.phi_x1) + log_y1
        if total_0 >= total_1:
            return 1
        else:
            return 0

    def convert_to_matrix_naive(self, data_list):
        list_of_words = [word for x in data_list for word in x[0]]
        list_of_words.append('NOTAWORD')
        tmp = dict(enumerate(list_of_words))
        pos = {}
        for item in tmp:
            pos[tmp[item]] = item
        self.V = len(set(tmp.keys()))
        X = np.zeros((len(data_list),self.V))
        y = np.zeros((len(data_list),1))
        for i in range(len(data_list)):
            point = data_list[i]
            y[i] = point[1]
            for word in point[0]:
                 X[i,pos[word]] = 1
        self.word_list = pos
        return (np.asmatrix(X),np.asmatrix(y))

    def convert_to_matrix_test(self, data_list):
        X = np.zeros((len(data_list), self.V))
        y = np.zeros((len(data_list), 1))
        for i in range(len(data_list)):
            point = data_list[i]
            y[i] = point[1]
            for word in point[0]:
                if word in self.word_list:
                    X[i, self.word_list[word]] = 1
                else:    
                    X[i, self.word_list['NOTAWORD']] = 1
        return (np.asmatrix(X), np.asmatrix(y))

