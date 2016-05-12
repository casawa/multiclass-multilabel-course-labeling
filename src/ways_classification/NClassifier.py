import Classifier
import LinearClassifier

class NClassifier(object):
    """Calls classification methods of the N classifiers we have"""
    def __init__(self, list_of_classifier_names, data_model):
        super(NClassifier, self).__init__()
        self.list_of_classifiers = []
        self.data_model = data_model
        for name in list_of_classifier_names:
            if name == "Naive Bayes":
                pass
            elif name == "Linear":
                self.list_of_classifiers.append(LinearClassifier())


    def train(self):
    """Calls the train method of each classifer and returns an array of the training errors"""
        training_errors = []
        for classifier in self.list_of_classifiers:
            training_errors.append(classifier.train(self.data_model.train_data))
        return training_errors

    def test(self):
    """Calls the test method of each classifer and returns an array of the test errors"""
        test_errors = []
        for classifer in self.list_of_classifiers:
            test_errors.append(classifier.test(self.data_model.test_data))
        return test_errors

    def classify(self, description):
    """Takes a single course description and returns an array of the training labels"""
        labels = []
        for classifer in self.list_of_classifiers:
            labels.append(classifier.classify(description))
        return labels
