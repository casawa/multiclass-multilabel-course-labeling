from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier
from sklearn.metrics import accuracy_score, roc_curve, auc
import util
import DataModel as dm
import numpy as np
from scipy import interp
import matplotlib.pyplot as plt

def construct_labels_matrix(data_model):
    Xt = None
    ymap = {}
    for way in data_model.get_list_of_ways():
        way_classes = set([tuple(x) for x in data_model.query_by_way(way,True)])
        all_classes = set([tuple(x[0]) for x in data_model.get_training_data()])
        tot = [(description,0) for description in all_classes]
        for i in range(len(tot)):
            elem = tot[i]
            desc = elem[0]
            if desc in way_classes:
                tot[i] = (desc,1)
        data_list = tot
        list_of_words = [word for x in data_list for word in x[0]]
        list_of_words.append("UNK")
        list_of_words = list(set(list_of_words))
        tmp = dict(enumerate(list_of_words))
        X,y = util.convert_to_matrix(data_list,tmp)
        Xt = X
        ymap[way] = y
    num_keys = len(ymap.keys())
    num_vals = len(ymap[ymap.keys()[0]])
    y = np.asmatrix(np.zeros((num_keys,num_vals)))
    for i in range(num_keys):
        for j in range(num_vals):
            y[i,j] = ymap[ymap.keys()[i]][j]
    y = y.transpose()
    return (X,y,tmp)

def construct_test_labels_matrix(data_model,temp):
    Xt = None
    ymap = {}
    for way in data_model.get_list_of_ways():
        way_classes = set([tuple(x) for x in data_model.query_by_way(way,False)])
        all_classes = set([tuple(x[0]) for x in data_model.get_testing_data()])
        tot = [(description,0) for description in all_classes]
        for i in range(len(tot)):
            elem = tot[i]
            desc = elem[0]
            if desc in way_classes:
                tot[i] = (desc,1)
        for i in range(len(tot)):
            elem = tot[i]
            desc = elem[0]
            desc = list(desc)
            for j in range(len(desc)):
                if desc[j] not in temp:
                    desc[j] = "UNK"
            desc = tuple(desc)
            tot[i] = (desc,elem[1])
        data_list = tot
        X,y = util.convert_to_matrix(data_list,temp)
        Xt = X
        ymap[way] = y
    num_keys = len(ymap.keys())
    num_vals = len(ymap[ymap.keys()[0]])
    y = np.asmatrix(np.zeros((num_keys,num_vals)))
    for i in range(num_keys):
        for j in range(num_vals):
            y[i,j] = ymap[ymap.keys()[i]][j]
    y = y.transpose()
    return (X,y)

def plot_ROC(X,y,newX,newy,clf):
    n_classes = y.shape[1]
    y_score = clf.decision_function(newX)
    print newy.ravel().shape
    print y_score.ravel().shape

    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(newy[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    k = y_score.ravel().shape[0]
    newy_rav = newy.ravel()
    newy_rav = newy_rav.reshape((k,1))
    y_score_rav = y_score.ravel()
    y_score_rav = y_score_rav.reshape((k,1))
    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(newy_rav, y_score_rav)
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])
    # Finally average it and compute AUC
    mean_tpr /= n_classes
    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
    # Plot all ROC curves
    plt.figure()
    plt.plot(fpr["micro"], tpr["micro"],
             label='micro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["micro"]),
             linewidth=2)
    plt.plot(fpr["macro"], tpr["macro"],
             label='macro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["macro"]),
             linewidth=2)

    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Micro- and Macro-Average ROC curves')
    plt.legend(loc="lower right")
    plt.show()

def main():
    data_model = dm.DataModel()
    #X,y,tmp = construct_labels_matrix(data_model)
    #newX,newy = construct_test_labels_matrix(data_model,tmp)
    #clf = OneVsRestClassifier(SVC(kernel='linear'))
    #clf.fit(X,y)
    #train_pred = clf.predict(X)
    #test_pred = clf.predict(newX)
    #print "Training error linear: " + str(1.0 - accuracy_score(train_pred, y))
    #print "Testing error linear: " + str(1.0 - accuracy_score(test_pred, newy))
    #plot_ROC(X,y,newX,newy,clf)


    X,y,tmp = construct_labels_matrix(data_model)
    newX,newy = construct_test_labels_matrix(data_model,tmp)
    clf2 = OneVsRestClassifier(SVC(kernel='rbf'))
    clf2.fit(X,y)
    train_pred = clf2.predict(X)
    test_pred = clf2.predict(newX)
    print "Training error rbf: " + str(1.0 - accuracy_score(train_pred, y))
    print "Testing error rbf: " + str(1.0 - accuracy_score(test_pred, newy))
    plot_ROC(X,y,newX,newy,clf2)

if __name__ == '__main__':
    main()
