### Defines generalized Classifier class
class Classifier(object):
    def __init__(self, data_model, way):
        self.data_model = data_model
        self.way = way

    def train(self): pass

    def test(self): pass

    def classify(self): pass
