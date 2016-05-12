### Defines Linear Classifier class
import Classifier
import numpy as np
from abc import ABCMeta, abstractmethod

class LinearClassifier(Classifier):
    """Represents a linear classifier"""
    def __init__(self, data_model, way):
        self.data_model = data_model
        self.way = way
        self.weights = None

    def train(self):
        """Trains model on data from data_model"""
        way_classes = dict(self.data_model.get_train_courses(way))
        all_classes = dict(self.data_model.get_all_train_courses_with_ways())
        neg = [(description,0) for key in all_classes.keys() if key not in way_classes.keys()]
        pos = [(description,1) for key in all_classes.keys() if key not in way_classes.keys()]
        data_list = pos + neg
        X,y = convert_to_matrix(data_list)
        self.weights = np.linalg.inv(X.transpose()*X)*X.transpose*y
        return sum(np.square(X*self.weights - y))

    def test(self):
        """Tests model on data from data_model"""
        way_classes = dict(self.data_model.get_test_courses(way))
        all_classes = dict(self.data_model.get_all_test_courses_with_ways())
        neg = [(description,0) for key in all_classes.keys() if key not in way_classes.keys()]
        pos = [(description,1) for key in all_classes.keys() if key not in way_classes.keys()]
        data_list = pos + neg
        X,y = convert_to_matrix(data_list)
        return sum(np.square(X*self.weights - y))
