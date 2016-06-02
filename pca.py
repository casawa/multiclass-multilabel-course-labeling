import numpy as np
import matplotlib.pyplot as plt

l = [10,50,100,250,500,750,1000]
h = [0.685126582278,0.445675105485,0.446202531646,0.431434599156,0.45253164557,0.407172995781,0.425105485232]
t = [0.15585443038,0.111550632911,0.107594936709,0.0996835443038,0.102848101266,0.0996835443038,0.102056962025]

a, = plt.plot(l,h,"r",label="Hamming Error")
b, = plt.plot(l,t,"b",label="Average Test Error")
plt.legend(handles=[a,b])
plt.title("Error by number of PCA components")
plt.xlabel("Number of PCA components")
plt.ylabel("Error")
plt.show()
