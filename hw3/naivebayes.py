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
    n = len(Ytrain)
    n0 = np.count_nonzero(Ytrain)
    n1 = n - n0
    
    return n0/n, n1/n


def get_x_given_y(Xtrain, Ytrain, M):
    n = len(Ytrain)
    x_given_y0 = np.zeros((M,)).astype(int)
    x_given_y1 = np.zeros((M,)).astype(int)
    
    
    for i in range(n):
        y = Ytrain[i]
        for j in range(M):
            if y==0:
                x_given_y0[j]+= Xtrain[i, j]
            else:
                x_given_y1[j]+= Xtrain[i, j]
                
    
    return x_given_y0, x_given_y1


def predict_on_train(Xtrain, prior_y0, prior_y1, x_given_y0, x_given_y1):
    n = len(Xtrain)
    M = len(x_given_y0)
    y_pred = np.zeros((n, ))
    for i in range(n):
        prod0 = 1
        prod1 = 1
        for j in range(M):
            xj = Xtrain[i][j] 
            if xj == 1: #we will only consider the words present in this sentence    
                prod0 = prod0* ((x_given_y0[j]+1)/(n+2))  ##laplace smoothing add 1 to the numerator, 2 to the denominator
                prod1 = prod1* ((x_given_y1[j]+1)/(n+2))
        y0_given_x = prior_y0*prod0
        y1_given_x = prior_y1*prod1
        
        if y1_given_x >= y0_given_x:
            y_pred[i] = 1
        else:
            y_pred[i] = 0
            
            
    return y_pred.astype(int)

        
    
    
def predict_on_test(Xtest, prior_y0, prior_y1, x_given_y0, x_given_y1, n):
    M = len(x_given_y0)
    ntest = len(Xtest)
    y_pred = np.zeros((ntest, ))
    for i in range(ntest):
        prod0 = 1
        prod1 = 1
        for j in range(M):
            xj = Xtest[i][j] 
            if xj == 1: #we will only consider the words present in this sentence    
                prod0 = prod0* ((x_given_y0[j]+1)/(x_given_y0[j]+x_given_y1[j]+2))  ##laplace smoothing add 1 to the numerator, 2 to the denominator
                prod1 = prod1* ((x_given_y1[j]+1)/(x_given_y0[j]+x_given_y1[j]+2))
                #for totally unseen data, it will multiply half with both products, so we can safely ignore them
        y0_given_x = prior_y0*prod0
        y1_given_x = prior_y1*prod1
        
        if y1_given_x >= y0_given_x:
            y_pred[i] = 1
        else:
            y_pred[i] = 0
            
    
    return y_pred.astype(int)
            
    
    
    
    
    
    