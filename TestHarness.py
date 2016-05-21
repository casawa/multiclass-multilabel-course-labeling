from Classifier import Classifier
import LinearClassifier as lc
import NaiveBayes as nb
import DataModel as dm
import matplotlib.pyplot as plt
import numpy as np
import random

def test_linear():
    '''This is a wrapper around the test harness for the linear classifier.'''
    data = dm.DataModel()
    list_of_ways = data.get_list_of_ways()
    results = {}
    for way in list_of_ways:
        clf = lc.LinearClassifier(data, way)
        results[way] = _test(clf, way)
    ax = _show_results(results)
    ax.set_title('Linear Classifier Per-Category Error')
    plt.show()


def test_NB():
    '''This is a wrapper aroung the test harness for the Naive Bayes classifier'''
    data = dm.DataModel()
    list_of_ways = data.get_list_of_ways()
    results = {}
    for way in list_of_ways:
        clf = nb.NaiveBayes(data, way)
        results[way] = _test(clf, way)
    ax = _show_results(results)
    ax.set_title('Naive Bayes Classifier Per-Category Error')
    plt.show()


def _test(classifier, way):
    '''The test harness. Takes a classifer and prints statistics'''
    print way
    train_error = classifier.train()
    test_error = classifier.test()
    print 'Training error: {}'.format(train_error)
    print 'Test error: {}'.format(test_error)
    return (train_error, test_error)



def _show_results(results):
    tests = []
    trains = []
    labels = []
    for way, result in results.iteritems():
        labels.append(way)
        trains.append(result[0])
        tests.append(result[1])
    fig, ax = plt.subplots()
    N = len(labels)
    width = 0.35
    ind = np.arange(N)
    rects1 = ax.bar(ind, trains, width, color='r')
    rects2 = ax.bar(ind + width, tests, width, color='b')
    ax.set_ylabel('Errors')
    ax.set_xticks(ind+width)
    ax.set_xticklabels(tuple(labels))
    ax.legend((rects1[0], rects2[0]), ('Training', 'Testing'))
    #for rect in rects1:
    #   height = rect.get_height()
    #    ax.text(rect.get_x() + rect.get_width()/2., height + 0.05, '%.3f' % height,ha='center',va='bottom')
    #for rect in rects2:
    #    height = rect.get_height()
    #    ax.text(rect.get_x() + rect.get_width()/2., height + 0.05, '%.3f' % height,ha='center',va='bottom')
    return ax

"""
Determines the WAYS for a particular course description.
"""
# Data is an instance of DataModel
def overall_naive_bayes_test(data, course_desc):
    list_of_ways = data.get_list_of_ways()
    total_dist = 0
    ways_to_classifiers = {}
    for way in list_of_ways:
        clf = nb.NaiveBayes(data, way)
        clf.train()
        ways_to_classifiers[way] = clf

    for test_ex in data.training_data:
        test_way = test_ex[1]
        course_desc = test_ex[0]

        clf = ways_to_classifiers[test_way]
        result = clf.classify(course_desc)
        if result == 0:
            total_dist += 1
    print float(total_dist)/len(data.training_data)

    total_dist = 0
    for test_ex in data.testing_data:
<<<<<<< HEAD
        test_ways = text_ex[1]
        course_desc = text_ex[0]

        predicted_ways = []
        for way in ways_to_classifiers:
            clf = ways_to_classifiers[way]
            result = clf.classify(course_desc)
            if result == 1:
                predicted_ways.append(way)

# Data is an instance of DataModel
def overall_linear_test(data, course_desc):
    list_of_ways = data.get_list_of_ways()

    ways_to_classifiers = {}
    for way in list_of_ways:
        clf = lc.LinearClassifier(data, way)
        clf.train()
        ways_to_classifiers[way] = clf

    for test_ex in data.testing_data:
        test_ways = test_ex[1]
        course_desc = test_ex[0]

        predicted_ways = []
        for way in ways_to_classifiers:
            clf = ways_to_classifiers[way]
            result = clf.classify(course_desc)
            if result == 1:
                predicted_ways.append(way)

def main():
    data = dm.DataModel()
    #print overall_naive_bayes_test(data, ['numbers', 'modeling', 'mathematical'])
    #print overall_linear_test(data,['numbers','modeling','mathematical'])
    test_linear()
    
if __name__ == '__main__':
    main()
