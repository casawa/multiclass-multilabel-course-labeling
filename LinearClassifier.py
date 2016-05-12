### Defines Linear Classifier class
from Classifier import Classifier
import numpy as np
import util

class LinearClassifier(Classifier):
    """Represents a linear classifier"""
    def __init__(self, data_model, way):
        super(LinearClassifier, self).__init__(data_model,way)
        self.weights = None

    def train(self):
        """Trains model on data from data_model"""
        way_classes = dict(self.data_model.query_by_way(way,True))
        all_classes = dict(self.data_model.get_training_data())
        neg = [(description,0) for description in all_classes.keys() if description not in way_classes.keys()]
        pos = [(description,1) for description in all_classes.keys() if description not in way_classes.keys()]
        data_list = pos + neg
        X,y = convert_to_matrix(data_list)
        self.weights = np.linalg.inv(X.transpose()*X)*X.transpose()*y
        err = float(sum(np.sign(np.abs(np.sign(X*self.weights) - y))))/y.shape[0]
        return err

    def test(self):
        """Tests model on data from data_model"""
        way_classes = dict(self.data_model.query_by_way(way,False))
        all_classes = dict(self.data_model.get_testing_data())
        neg = [(description,0) for description in all_classes.keys() if description not in way_classes.keys()]
        pos = [(description,1) for description in all_classes.keys() if description not in way_classes.keys()]
        data_list = pos + neg
        X,y = convert_to_matrix(data_list)
        X,y = self.get_temp_train_data()
        err = float(sum(np.sign(np.abs(np.sign(X*self.weights) - y))))/y.shape[0]
        return err
