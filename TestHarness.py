from Classifier import Classifier
import LinearClassifier as lc
import NaiveBayes as nb
import DataModel as dm

def test_linear():
    '''This is a wrapper around the test harness for the linear classifier.'''
    data = dm.DataModel()
    list_of_ways = data.get_list_of_ways()
    for way in list_of_ways:
        clf = lc.LinearClassifier(data, way)
        _test(clf, way)


def test_NB():
    '''This is a wrapper aroung the test harness for the Naive Bayes classifier'''
    data = dm.DataModel()
    list_of_ways = data.get_list_of_ways()
    for way in list_of_ways:
        clf = nb.NaiveBayes(data, way)
        _test(clf, way)
  
def _test(classifier, way):
    '''The test harness. Takes a classifer and prints statistics'''
    print way
    print 'Training error: {}'.format(classifier.train())
    print 'Test error: {}'.format(classifier.test())

