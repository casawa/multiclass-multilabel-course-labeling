### Defines Linear Classifier class
import Classifier
from abc import ABCMeta, abstractmethod

class LinearClassifier(Classifier):
    def __init__(self, data_model,way):
        self.data_model = data_model
        self.way = way
        self.weights = None

    def train(self):
        way_classes = dict(self.data_model.get_courses(way))
        all_classes = dict(self.data_model.get_all_courses_with_ways())
        neg = [(description,0) for key in all_classes.keys() if key not in way_classes.keys()]
        pos = [(description,1) for key in all_classes.keys() if key not in way_classes.keys()]
        data_list = pos + neg
        X,y = convert_to_matrix(data_list)
