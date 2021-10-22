# -*- coding: utf-8 -*-
"""
@author: Abdullah Mamun
"""
import numpy as np
from sklearn.utils import shuffle
from copy import deepcopy
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def SVM_tuning(Xtrain, Ytrain, Xtest, Ytest, problemNo):
    n = Xtrain.shape[0]
    #print(Xtrain.shape)
    #print(n)
    last = int(0.8*n) #using first 80% as training and the rest as validation
    #print(last)
    Xtr = Xtrain[:last]
    Ytr = Ytrain[:last]
    Xval = Xtrain[last:]
    Yval = Ytrain[last:]
    
    trlen = len(Ytr)  #length of new training set
    vallen = len(Yval) #length of new validation set
    testlen = len(Ytest) #length of test set
    print(Xtr.shape, Ytr.shape)
    
    clf = make_pipeline(StandardScaler(), SVC(C=10**(4), random_state=0))
    
    clf.fit(Xtr, Ytr)
    #print(clf.named_steps['linearsvc'].coef_)
    #print(clf.named_steps['linearsvc'].intercept_)
    

    trainacc = (np.count_nonzero(clf.predict(Xtr)==Ytr))/trlen
    valacc = (np.count_nonzero(clf.predict(Xval)==Yval))/vallen
    testacc = (np.count_nonzero(clf.predict(Xtest)==Ytest))/testlen
    print(trainacc, valacc, testacc)
    print(len(clf.named_steps['svc'].support_vectors_))

def plot_graph(x, y, xlabel, ylabel, title):
    plt.plot(x, y)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
     
    plt.title(title)
    try:
        plt.savefig('{}.png'.format(title.replace('.','_')))
    except:
        pass
    plt.show()
    
        