### Defines Linear Classifier class
from Classifier import Classifier
from sklearn import linear_model
import numpy as np
import util
import sys
import cvxpy as cvx

class RANKSVM(Classifier):
    """Represents a linear classifier"""
    def __init__(self, data_model, way):
        super(RANKSVM, self).__init__(data_model,way)
        self.data_model = data_model
        self.classifier = None
        self.list_of_words = None
        self.tmp = None

    def train(self):
        """Trains model on data from data_model"""
        classes = self.data_model.get_list_of_ways()
        num_classes = len(classes)
        cmap = {}
        for i in range(len(classes)):
            cmap[classes[i]] = i
        training_data = self.data_model.training_data_all_ways
        train_unform = [(description[0],cmap[description[1][0]]) for description in training_data]
        list_of_words = [word for x in train_unform for word in x[0]]
        list_of_words.append("UNK")
        list_of_words = list(set(list_of_words))
        tmp = dict(enumerate(list_of_words))
        self.list_of_words = list_of_words
        self.tmp = tmp
        X,y = util.convert_to_PCA_matrix(train_unform,tmp,10)

        Y = np.asmatrix(np.zeros((len(training_data),num_classes)))
        Ybar = np.asmatrix(np.zeros((len(training_data),num_classes)))
        for i in range(len(training_data)):

            w = training_data[i][1]
            for way in w:
                Y[i,cmap[way]] = 1
        for i in range(Y.shape[0]):
            for j in range(Y.shape[1]):
                if Y[i,j] == 1:
                    Ybar[i,j] = 0
                else:
                    Ybar[i,j] = 1

        s_alph = 0.0
        for i in range(Y.shape[0]):
            y = Y[i,:]
            ybar = Ybar[i,:]
            s_alph = s_alph + np.sum(y)*np.sum(ybar)
        print s_alph
        alpha = np.zeros((int(s_alph),1))
        print alpha.shape
        C = np.zeros((num_classes,int(s_alph)))
        for k in range(num_classes):
            for i in range(Y.shape[0]):
                for j in range(8):
                    for l in range(8):
                        if j != k and l != k: continue
                        if j == k:
                            C[k,i + 8*j + 8*l] = 1.0
                        else:
                            C[k,i + 8*j + 8*l] = -1.0

        Xdot = np.zeros((Y.shape[0],Y.shape[0]))
        print Xdot.shape
        for i in range(Y.shape[0]):
            print i
            for j in range(Y.shape[0]):
                xi = np.asmatrix(X[i,:])
                xj = np.asmatrix(X[j,:])
                Xdot[i,j] = float(np.dot(xi,xj.transpose()))

        while True:
            beta = np.asmatrix(np.zeros((len(training_data),num_classes)))
            print beta.shape[0]
            for k in range(beta.shape[1]):
                for i in range(beta.shape[0]):
                    y = np.asarray(Y[i,:])[0]
                    ybar = np.asarray(Ybar[i,:])[0]
                    yset = [p for p in range(beta.shape[1]) if y[p] == 1]
                    ybarset = [q for q in range(beta.shape[1]) if ybar[q] == 1]
                    tups = []
                    for p in yset:
                        for q in ybarset:
                            tups.append((p,q))
                    for tup in tups:
                        j = tup[0]
                        l = tup[1]
                        const = 0
                        if j != k and l != k: continue
                        if j == k:
                            const = 1
                        else:
                            const = -1
                        beta[i,j] = beta[i,k] + const*alpha[i + 8*j + 8*l]

            dots = np.asmatrix(np.zeros((len(training_data),num_classes)))
            for k in range(dots.shape[1]):
                for i in range(dots.shape[0]):
                    xi = X[i,:]
                    xj = X[j,:]
                    dots[i,k] = dots[i,k] + beta[i,k]*float(xi*xj.transpose())

            g = np.zeros((int(s_alph),1))
            for i in range(dots.shape[0]):
                for k in range(dots.shape[1]):
                    for l in range(dots.shape[1]):
                        g[i + 8*k + 8*l] = dots[i,k] - dots[i,l] - 1.0

            alpha_new = cvx.Variable(int(s_alph))
            obj = cvx.Minimize(alpha_new.T*g)
            constraints = [0 <= alpha_new, alpha_new <= 1.0/float(num_classes)]
            for k in range(num_classes):
                cvec = np.asmatrix(C[k,:])
                constraints.append(cvec*alpha_new == 0)
            prob = cvx.Problem(obj,constraints)
            prob.solve()
            print prob.status

            anew = alpha_new.value

            a2 = cvx.Variable(int(s_alph))
            lam = cvx.Variable()
            constraints = [0 <= a2, a2 <= 1.0/float(num_classes), 0 <= lam, lam <= 1]
            constraints.append(a2 == alpha + lam*anew)
            print "RUNNING OBJECTIVE"
            def obj_fun(BETA,XDOT,ALPHA,Ys,YBAR):
                res = 0.0
                for k in range(num_classes):
                    print "Class: " + str(k)
                    vpk = BETA[k,:]*X
                    res = res + cvx.quad_over_lin(vpk,1)
                res = -0.5*res
                for i in range(num_classes):
                    print "Class: " + str(i)
                    y = np.asarray(Ys[i,:])[0]
                    ybar = np.asarray(Ybar[i,:])[0]
                    yset = [p for p in range(Y.shape[1]) if y[p] == 1]
                    ybarset = [q for q in range(Y.shape[1]) if ybar[q] == 1]
                    tups = []
                    for p in yset:
                        for q in ybarset:
                            tups.append((p,q))
                    for tup in tups:
                        j = tup[0]
                        l = tup[1]
                        res = res + ALPHA[i + 8*j + 8*l]
                return res

            VT = cvx.Variable(num_classes,Y.shape[0])

            for i in range(Y.shape[0]):
                print "EXAMPLE: " + str(i)
                for k in range(num_classes):
                    print "CLASS: " + str(k)
                    cvec = np.asmatrix(C[k,:])
                    y = np.asarray(Y[i,:])[0]
                    ybar = np.asarray(Ybar[i,:])[0]
                    yset = [p for p in range(Y.shape[1]) if y[p] == 1]
                    ybarset = [q for q in range(Y.shape[1]) if ybar[q] == 1]
                    tups = []
                    val = 0.0
                    for p in yset:
                        for q in ybarset:
                            tups.append((p,q))
                    for tup in tups:
                        j = tup[0]
                        l = tup[1]
                        val = val + cvec*a2
                    VT[k,i] = val
            obj = cvx.Maximize(obj_fun(VT,Xdot,a2,Y,Ybar))
            print obj
            for k in range(num_classes):
                cvec = np.asmatrix(C[k,:])
                constraints.append(cvec*a2 == 0)
            print len(constraints)
            prob = cvx.Problem(obj,constraints)
            print "SOLVING"
            prob.solve(verbose = True)
            print prob.status
            break
        return 0


    def test(self):
        """Tests model on data from data_model"""
        way_classes = set([tuple(x) for x in self.data_model.query_by_way(self.way,False)])
        all_classes = set([tuple(x[0]) for x in self.data_model.get_testing_data()])
        neg = [(description,0) for description in all_classes if description not in way_classes]
        pos = [(description,1) for description in all_classes if description in way_classes]
        data_list = pos + neg
        new_list = []
        for elem in data_list:
            description = list(elem[0])
            label = elem[1]
            for i in range(len(description)):
                if description[i] not in self.list_of_words:
                    description[i] = "UNK"
            new_list.append((tuple(description),label))
        X = None
        y = None
        X,y = util.convert_to_matrix(new_list,TMP)
        print X
        ypred = self.classifier.predict(X)
        err = 0
        for i in range(len(new_list)):
            yi = float(y[i])
            ypi = float(ypred[i])
            if yi != ypi:
                err += 1
        print str(float(err)) + " " + str(len(data_list))
        return float(err)/float(len(new_list))

    def classify(self, description):
        for i in range(len(description)):
            if description[i] not in self.list_of_words:
                description[i] = "UNK"
        X,y = util.convert_to_matrix([(tuple(description),0)],self.tmp)
        return int(self.classifier.predict(X))
