### Defines generalized Classifier class
from abc import ABCMeta, abstractmethod

class Classifier:
    __metaclass__ = ABCMeta

    self.data_model = None
    @abstractmethod
    def train(self): pass

    @abstractmethod
    def test(self): pass

    @abstractmethod
    def classify(self): pass
