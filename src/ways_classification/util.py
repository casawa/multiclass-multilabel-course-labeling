import numpy as np

def convert_to_matrix(data_list):
    strings = [x[0] for x in data_list]
    strings = strings.join(" ")
    strings = strings.strip()
    l_strings = strings.split()
    s_strings = set(l_strings)
    V = len(s_strings)
    mat = np.asmatrix(np.zeros((len(data_list,V))))
    
