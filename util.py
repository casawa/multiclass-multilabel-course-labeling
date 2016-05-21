import numpy as np
import math, random

def convert_to_matrix(data_list,tmp):
    pos = {}
    for item in tmp:
        pos[tmp[item]] = item
    V = len(set(tmp.keys()))
    X = np.zeros((len(data_list),V))
    y = np.zeros((len(data_list),1))
    for i in range(len(data_list)):
        point = data_list[i]
        y[i] = point[1]
        for word in point[0]:
             X[i,pos[word]] = X[i,pos[word]] + 1
    return (np.asmatrix(X),np.asmatrix(y))

"""
Helper functions for computing stochastic gradient descent
"""
def dotProduct(v1, v2):
	common_nonzero_indices = [index for index in v1 if index in v2]
	return sum([v1[index]*v2[index] for index in common_nonzero_indices])

def increment(v1, scale, v2):
    for elem in v2:
        if elem in v1:
            v1[elem] += (scale * v2[elem])
        else:
            v1[elem] = (scale * v2[elem])

def evaluate(examples, classifier):
    error = 0
    for x, y in examples:
        if classifier(x) != y:
            error += 1
    return float(error)/len(examples)

"""
Wrapper function that allows for cache-ing of feature extractions
"""
cache = {}
def fe(featureExtractor,x):
    xstr = x.tostring()
    if xstr in cache:
        return cache[xstr]
    else:
        res = featureExtractor(x)
        cache[xstr] = res
        return res

"""
Helper method that computes the R2 statistic, or the coefficient of
determination, which gives an estimate of the amount of variance
explained by a given linear model. R2 = 1 - SSres/SStot, where
SStot = sum(yi - ybar)^2 and SSres = sum(predicted_i - yi)^2
"""
def computeR(trainExamples, featureExtractor, weights):
    mean = sum([x[1] for x in trainExamples])/float(len(trainExamples))
    print mean
    SStot = sum([math.pow(float(x[1] - mean),2) for x in trainExamples])
    pred = [dotProduct(fe(featureExtractor,x[0]),weights) for x in trainExamples]
    for i in range(0,len(pred)):
        if pred[i] > 0:
            pred[i] = 1.0
        else:
            pred[i] = -1.0
    SSres = sum([math.pow(float(trainExamples[i][1] - pred[i]),2) for i in range(0,len(trainExamples))])
    return float(1) - float(SSres)/SStot


"""
Function for computing a linear classifier using the method of stochasitc gradient descent. Generates model,
computes errors and relevant statistics, and writes them to an out file"
"""
def SGD(trainExamples, featureExtractor, testExamples = None, numIters=100, stepSize=0.00225, debug=True):
    weights = {}  # feature => weight
    features = {}
    def grad(weights, trainExample):
        x = trainExample[0]
        y = trainExample[1]
        features = fe(featureExtractor,x)
        if y*dotProduct(weights, features) < 1:
            for value in features:
                features[value] *= -y
            return features
        else:
            return {}


    print numIters
    for i in range(numIters):
        print "Iteration: " + str(i)
        random.shuffle(trainExamples)
        G = 0
        for trainExample in trainExamples:
            print G
            G += 1
            gradient = grad(weights, trainExample)
            print sum(gradient.values())
            step = float(1)/math.sqrt(i+1)
            increment(weights, -step, gradient)
        if debug:
            trainError = evaluate(trainExamples, lambda(x) : (1 if dotProduct(fe(featureExtractor,x), weights) >= 0 else -1))
            print 'Train error: ' + str(trainError)
    return (weights,trainError)
