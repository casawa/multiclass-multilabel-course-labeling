#Defines NaiveBayesClassifier Class
from Classifier import Classifier
import numpy as np
import util

class NaiveBayes(Classifier):
    def __init__(self, data_model, way):
        super(NaiveBayes, self).__init__(data_model,way)
        self.phi_y = 0
        self.phi_x0 = None
        self.phi_x1 = None


    def train(self):
        way_classes = dict(self.data_model.query_by_way(way,True))
        all_classes = dict(self.data_model.get_training_data())
        neg = [(description,0) for description in all_classes.keys() if key not in way_classes.keys()]
        pos = [(description,1) for description in all_classes.keys() if key not in way_classes.keys()]
        data_list = pos + neg
        X, y = convert_to_matrix(data_list)
        m = X.shape[1]
        self.phi_x0 = ones(X.shape[2], 1)
        self.phi_x1 = ones(X.shape[2], 1)
        for x in np.nditer(X, op_flags=['readwrite']):
            if x > 0:
                x = 1
        for index, row in enumerate(X):
            if y[index] == 0:
                self.phi_x0 += row
            else:
                self.phi_x1 += row
                self.phi_y += 1
        self.phi_x1 /= (phi_y + 2)
        self.phi_x0 /= (m + 2 - phi_y)
        self.phi_y /= m

        self.phi_x1 = [np.log(x) for x in self.phi_x1]
        self.phi_x0 = [np.log(x) for x in self.phi_x0]

        errors = 0
        for index, row in enumerate(X):
            if get_predicted_class(row) == y[index]:
                errors += 1
        errors /= m
        return errors

    def test(self):
        """Tests model on data from data_model"""
        way_classes = dict(self.data_model.query_by_way(way,False))
        all_classes = dict(self.data_model.get_testing_data())
        neg = [(description,0) for key in all_classes.keys() if key not in way_classes.keys()]
        pos = [(description,1) for key in all_classes.keys() if key not in way_classes.keys()]
        data_list = pos + neg
        m = shape(X)[1]
        X,y = convert_to_matrix(data_list)
        errors = 0
        for index, row in enumerate(X):
            if get_predicted_class(row) != y[index]:
                errors += 1
        return errors / m

    def classify(self, description):
        pass

    def get_predicted_class(x):
        total_0 = np.dot(self.phi_x0, x) + log(self.phi_y)
        total_1 = np.dot(self.phi_x1, x) + log(1 - self.phi_y)
        if total_0 >= total_1:
            return 1
        else:
            return 0
