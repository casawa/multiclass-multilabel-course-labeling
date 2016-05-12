### Defines generalized Classifier class
from abc import ABCMeta, abstractmethod

class Classifier:
    __metaclass__ = ABCMeta

    @abstractmethod
    def train(self): pass

    @abstractmethod
    def test(self): pass

    @abstractmethod
    def classify(self): pass
