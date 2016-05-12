### Defines Linear Classifier class
from Classifier import Classifier
import numpy as np
import util

class LinearClassifier(Classifier):
    """Represents a linear classifier"""
    def __init__(self, data_model, way):
        super(LinearClassifier, self).__init__(data_model,way)
        self.data_model = data_model
        self.way = way
        self.weights = None

    def train(self):
        """Trains model on data from data_model"""
        wc = self.data_model.query_by_way(self.way,True)
        al = self.data_model.get_training_data()
        way_classes = [tuple(elem[0]) for elem in wc]
        all_classes = [tuple(elem[0]) for elem in al]
        neg = [(description,0) for description in all_classes if description not in way_classes]
        pos = [(description,1) for description in all_classes if description not in way_classes]
        data_list = pos + neg
        X,y = util.convert_to_matrix(data_list)
        self.weights = np.linalg.inv(X.transpose()*X)*X.transpose()*y
        err = float(sum(np.sign(np.abs(np.sign(X*self.weights) - y))))/y.shape[0]
        return err

    def test(self):
        """Tests model on data from data_model"""
        way_classes = dict(self.data_model.query_by_way(way,False))
        all_classes = dict(self.data_model.get_testing_data())
        neg = [(description,0) for description in all_classes if description not in way_classes]
        pos = [(description,1) for description in all_classes if description not in way_classes]
        data_list = pos + neg
        X,y = convert_to_matrix(data_list)
        X,y = self.get_temp_train_data()
        err = float(sum(np.sign(np.abs(np.sign(X*self.weights) - y))))/y.shape[0]
        return err
