from Classifier import Classifier
import LinearClassifier as lc
import LinearPCAClassifier as lpc
import NaiveBayes as nb
import DataModel as dm
import matplotlib.pyplot as plt
import numpy as np
import random
import threading


def test_linear():
    '''This is a wrapper aroung the test harness for the Naive Bayes classifier'''
    data = dm.DataModel()
    list_of_ways = data.get_list_of_ways()
    results = {}
    numiter = 20
    for way in list_of_ways:
        clf = lc.LinearClassifier(data, way)
        if way in results:
            tr = _test(clf, way)
            results[way] = (results[way][0] + tr[0], results[way][1] + tr[1])
        else:
            results[way] = _test(clf, way)

    print results
    ax = _show_results(results)
    ax.set_title('Linear Classifier Per-Category Error')
    plt.show()

def test_NB():
    '''This is a wrapper aroung the test harness for the Naive Bayes classifier'''
    data = dm.DataModel()
    list_of_ways = data.get_list_of_ways()
    results = {}
    numiter = 20
    for way in list_of_ways:
        clf = nb.NaiveBayes(data, way)
        if way in results:
            tr = _test(clf, way)
            results[way] = (results[way][0] + tr[0], results[way][1] + tr[1])
        else:
            results[way] = _test(clf, way)

    print results
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

# Data is an instance of DataModel
def overall_naive_bayes_test(data):
    print "Naive Bayes False Positives"
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
    print "Training Error:", float(total_dist)/len(data.training_data)

    total_dist = 0
    for test_ex in data.testing_data:
        test_way = test_ex[1]
        course_desc = test_ex[0]

        clf = ways_to_classifiers[test_way]
        result = clf.classify(course_desc)
        if result == 0:
            total_dist += 1
    print "Testing Error:", float(total_dist)/len(data.testing_data)


"""
Determines the Hamming distance error for Naive Bayes.
"""
# Data is an instance of DataModel
def overall_naive_bayes_ham_test(data):

    print "Naive Bayes Hamming Distance"
    list_of_ways = data.get_list_of_ways()
    total_dist = 0
    ways_to_classifiers = {}
    for way in list_of_ways:
        clf = nb.NaiveBayes(data, way)
        clf.train()
        ways_to_classifiers[way] = clf

    for test_ex in data.testing_data_all_ways:
        test_ways = set(test_ex[1])
        course_desc = test_ex[0]

        predicted_ways = set()
        for way in ways_to_classifiers:
            clf = ways_to_classifiers[way]
            result = clf.classify(course_desc)
            if result == 1:
                predicted_ways.add(way)
        predicted_ways = set(predicted_ways)

    print total_dist/len(data.testing_data_all_ways)

"""
Linear test false positives
"""
# Data is an instance of DataModel
def overall_linear_test(data):

    print "Linear Test False Positives"
    list_of_ways = data.get_list_of_ways()

    total_dist = 0
    ways_to_classifiers = {}

    for way in list_of_ways:
        clf = lc.LinearClassifier(data, way)
        err[way] = (clf.train(),clf.test())
        print way + " train " + str(err[way][0])
        print way + " test " + str(err[way][1])
        ways_to_classifiers[way] = clf

    for test_ex in data.training_data:
        test_way = test_ex[1]
        course_desc = test_ex[0]

        clf = ways_to_classifiers[test_way]
        result = clf.classify(course_desc)
        if result == 0:
            total_dist += 1
    print "Training Error: ", float(total_dist)/len(data.training_data)

    total_dist = 0
    for test_ex in data.testing_data:
        test_way = test_ex[1]
        course_desc = test_ex[0]

        clf = ways_to_classifiers[test_way]
        result = clf.classify(course_desc)
        if result == 0:
            total_dist += 1
    print "Testing Error: ", float(total_dist)/len(data.testing_data)


"""
Hamming distance for linear classifier.
"""
# Data is an instance of DataModel
def overall_linear_ham_test(data):

    print "Linear hamming distance"
    list_of_ways = data.get_list_of_ways()
    total_dist = 0
    ways_to_classifiers = {}
    err = {}
    for way in list_of_ways:
        clf = lc.LinearClassifier(data, way)
        err[way] = (clf.train(),clf.test())
        print way + " train " + str(err[way][0])
        print way + " test " + str(err[way][1])
        ways_to_classifiers[way] = clf

    for test_ex in data.testing_data_all_ways:
        test_ways = set(test_ex[1])
        course_desc = test_ex[0]

        predicted_ways = set()
        for way in ways_to_classifiers:
            clf = ways_to_classifiers[way]
            result = clf.classify(course_desc)
            if result == 1:
                predicted_ways.add(way)
        #print test_ways
        total_dist += 1 - float(len((predicted_ways & test_ways)))/len(predicted_ways | test_ways)

    print total_dist/len(data.testing_data_all_ways)
    avg_test = sum([err[way][1]for way in err])/len(err.keys())
    print "Avg test: " + str(avg_test)


"""
Linear PCA test false positives
"""
# Data is an instance of DataModel
def overall_linear_pca_test(data):

    print "Linear PCA Test False Positives"
    list_of_ways = data.get_list_of_ways()

    total_dist = 0
    ways_to_classifiers = {}

    for way in list_of_ways:
        print way
        clf = lpc.LinearPCAClassifier(data, way, 10)
        clf.train()
        ways_to_classifiers[way] = clf

    for test_ex in data.training_data:
        test_way = test_ex[1]
        course_desc = test_ex[0]

        clf = ways_to_classifiers[test_way]
        result = clf.classify(course_desc)
        if result == 0:
            total_dist += 1
    print "Training Error: ", float(total_dist)/len(data.training_data)

    total_dist = 0
    for test_ex in data.testing_data:
        test_way = test_ex[1]
        course_desc = test_ex[0]

        clf = ways_to_classifiers[test_way]
        result = clf.classify(course_desc)
        if result == 0:
            total_dist += 1
    print "Testing Error: ", float(total_dist)/len(data.testing_data)


exitFlag = 0
queueLock = threading.Lock()
threads = []
NPOS = [3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]

def pca_func(data):
    global NPOS
    n = -1
    while not exitFlag:
        queueLock.acquire()
        if len(NPOS) != 0:
            n = NPOS[0]
            NPOS = NPOS[1:]
            queueLock.release()
            print "Running component " + str(n)
        else:
            queueLock.release()

        list_of_ways = data.get_list_of_ways()
        total_dist = 0
        ways_to_classifiers = {}
        for way in list_of_ways:
            clf = lpc.LinearPCAClassifier(data, way, n)
            clf.train()
            ways_to_classifiers[way] = clf

        for test_ex in data.testing_data_all_ways:
            test_ways = set(test_ex[1])
            course_desc = test_ex[0]

            predicted_ways = set()
            for way in ways_to_classifiers:
                clf = ways_to_classifiers[way]
                result = clf.classify(course_desc)
                if result == 1:
                    predicted_ways.add(way)
            #print test_ways
            total_dist += 1 - float(len((predicted_ways & test_ways)))/len(predicted_ways | test_ways)

        ham =  total_dist/len(data.testing_data_all_ways)
        print "Components: " + str(n) + ", Error: " + str(ham)

import Queue
class myThread (threading.Thread):
    def __init__(self, threadID, data):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.data = data
    def run(self):
        print "Starting " + str(self.threadID)
        pca_func(self.data)
        print "Exiting " + str(self.threadID)

"""
Hamming distance for linear PCA classifier.
"""
# Data is an instance of DataModel
def overall_linear_pca_ham_test_parallel(data):

    print "Linear hamming distance"
    hamming = {}
    for i in range(3):
        thread_n = myThread(i,data)
        threads.append(thread_n)
    for thread in threads: thread.start()


    while True:
        queueLock.acquire()
        if len(NPOS) != 0:
            queueLock.release()
            pass
        else:
            queueLock.release()
            break

    exitFlag = 1

def overall_linear_pca_ham_test(data):
    list_of_ways = data.get_list_of_ways()
    total_dist = 0
    ways_to_classifiers = {}
    err = {}
    for way in list_of_ways:
        clf = lpc.LinearPCAClassifier(data, way, 100)
        err[way] = (clf.train(),clf.test())
        print way + " train " + str(err[way][0])
        print way + " test " + str(err[way][1])
        ways_to_classifiers[way] = clf

    for test_ex in data.testing_data_all_ways:
        test_ways = set(test_ex[1])
        course_desc = test_ex[0]

        predicted_ways = set()
        for way in ways_to_classifiers:
            clf = ways_to_classifiers[way]
            result = clf.classify(course_desc)
            if result == 1:
                predicted_ways.add(way)
        #print test_ways
        total_dist += 1 - float(len((predicted_ways & test_ways)))/len(predicted_ways | test_ways)

    ham =  total_dist/len(data.testing_data_all_ways)
    print "Components: " + str(100) + ", Error: " + str(ham)
    avg_test = sum([err[way][1]for way in err])/len(err.keys())
    print "Avg test: " + str(avg_test)


def main():
    data = dm.DataModel()
    #print overall_naive_bayes_test(data, ['numbers', 'modeling', 'mathematical'])
    #overall_linear_ham_test(data,['numbers','modeling','mathematical'])
    #test_linear()
    #test_linear()
    #overall_linear_test(data)
    for i in range(10): overall_linear_ham_test(data)
    #overall_linear_pca_test(data)
    for i in range(10): overall_linear_pca_ham_test(data)
    #overall_naive_bayes_test(data)
    #overall_naive_bayes_ham_test(data)
if __name__ == '__main__':
    main()
