import Classifier
import LinearClassifier

class NClassifier(object):
    """Calls classification methods of the N classifiers we have"""
    def __init__(self, classifier_type, data_model):
        super(NClassifier, self).__init__()
        self.list_of_classifiers = []
        self.data = data_model
        for way in self.data.get_list_of_ways():
            if classifier_type == "Naive Bayes":
                pass
            elif classifier_type == "Linear":
                self.list_of_classifiers.append(LinearClassifier(data_model, way))


    def train(self):
    """Calls the train method of each classifer and returns an array of the training errors"""
        training_errors = []
        for classifier in self.list_of_classifiers:
            training_errors.append(classifier.train())
        return train_errors

    def test(self):
    """Calls the test method of each classifer and returns an array of the test errors"""
        test_errors = []
        for classifer in self.list_of_classifiers:
            test_errors.append(classifier.test())
        return test_errors

    def classify(self, description):
    """Takes a single course description and returns an array of the training labels"""
        labels = []
        for index, classifer in enumerate(self.list_of_classifiers):
            if classifier.classify(description) == 1:
                labels.append(self.data.get_list_of_ways()[index]))
        return labels
