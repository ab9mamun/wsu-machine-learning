# -*- coding: utf-8 -*-
"""
@author: Abdullah Mamun
"""
import numpy as np
from sklearn.utils import shuffle
from copy import deepcopy
import matplotlib.pyplot as plt


def plot_graph(x, y, xlabel, ylabel, title):
    plt.semilogx(x, y)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
     
    plt.title(title)
    try:
        plt.savefig('{}.png'.format(title.replace('.','_')))
    except:
        pass
    plt.show()
    
    
    
    
def get_priors(Ytrain):
    n = len(ytrain)
    n0 = np.count_nonzero(ytrain)
    n1 = n - n0
    
    return n0/n, n1/n
