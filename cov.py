import DataModel as dm
import numpy as np
from sklearn.covariance import empirical_covariance
import matplotlib.pyplot as plt
import math
import seaborn as sns

def get_cov(data):
    dat = data.training_data_all_ways + data.testing_data_all_ways
    num_ways = len(data.get_list_of_ways())
    m = {}
    i = 0
    for way in data.get_list_of_ways():
        m[way] = i
        i += 1
    mat = np.zeros((num_ways,num_ways))
    for elem in dat:
        ways = elem[1]
        for way in ways:
            mat[m[way],m[way]] = mat[m[way],m[way]] + 1
        for w1 in ways:
            for w2 in ways:
                if w1 == w2: continue
                mat[m[w1],m[w2]] = mat[m[w1],m[w2]] + 1
    print mat
    emp_cov = empirical_covariance(mat)
    print emp_cov
    corr = np.zeros((num_ways,num_ways))
    for i in range(num_ways):
        for j in range(num_ways):
            corr[i,j] = emp_cov[i,j]/(math.sqrt(emp_cov[i,i])*math.sqrt(emp_cov[j,j]))
    print corr
    sns.heatmap(corr,vmin = -1, vmax = 1,square=True,xticklabels=m.keys(),yticklabels=m.keys())
    sns.plt.title("Covariance of WAYS frequencies")
    sns.plt.show()

def main():
    data = dm.DataModel()
    get_cov(data)

if __name__ == '__main__':
    main()
