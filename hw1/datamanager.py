# -*- coding: utf-8 -*-
"""
@author: Abdullah Mamun
"""
import pandas as pd
import numpy as np

def read_data():
    traindf = pd.read_csv('data/fashion-mnist_train.csv')
    testdf = pd.read_csv('data/fashion-mnist_test.csv')
    return traindf, testdf


def readAndPreprocess(dryrun=False, classifier=None):
    traindf, testdf = read_data()
        

    #print(traindf.head())
    #print(testdf.head())
    
    train, test = traindf.values, testdf.values
    if dryrun: # it is just to check if every function is working. but generally it will be false.
        train = train[:200,:]
        test = test[:100,:]
    Ytrain = train[:,0].astype(int)
    Xtrain = train[:,1:]
    Ytest = test[:,0].astype(int)
    Xtest = test[:, 1:]
    
    #scaling the pixel values between 0 and 1
    Xtrain = Xtrain/255.0
    Xtest = Xtest/255.0
    
    #if classifier == "binary":
    Ytrain_binary = Ytrain%2
    Ytest_binary = Ytest%2
    Ytrain_binary[Ytrain_binary==0] = -1
    Ytest_binary[Ytest_binary==0]= -1
    
    #np.savetxt("multilabels.csv", Ytrain, delimiter=",")
    #np.savetxt("binlabels.csv", Ytrain_binary, delimiter=",")
    
    return Xtrain, Ytrain, Xtest, Ytest, Ytrain_binary, Ytest_binary

