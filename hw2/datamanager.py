# -*- coding: utf-8 -*-
"""
@author: Abdullah Mamun
"""
import pandas as pd
import numpy as np

def read_dataFashion():
    traindf = pd.read_csv('data/fashion-mnist_train.csv')
    testdf = pd.read_csv('data/fashion-mnist_test.csv')
    return traindf, testdf


def readAndPreprocessFashion(dryrun=False, classifier=None):
    traindf, testdf = read_dataFashion()
        

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


def readAndPreprocessCancer(dryrun=False, classifier=None):
    df = pd.read_csv('data/breast-cancer-wisconsin.data', header=None)
    print('=== Breast-Cancer Dataset loaded===')
    #print(df.head())
    #print(df.tail())
    df.replace('?', 5, inplace=True)  #assigning some value. trivial imputation
    data = df.values
    X = data[:,1:10]
    Y = data[:,10]  #10th column
    Y = (Y-2)/2  #convert 2 and 4 to 0 and 1 respectively
    X = X.astype(float)
    Y = Y.astype(int)
    
    print("First 6 rows of features")
    print(X[:6,:])
    print("First 6 rows of labels (0=benign, 1=malignant)")
    print(Y[:6])
    
    print()
    print("Total examples:",len(Y),"Benign=",len(Y)-np.count_nonzero(Y), "Malignant=",np.count_nonzero(Y) )
    #print(txt)
    if dryrun:
        X, Y = X[:200], Y[:200]
        
    n = len(X)
    pivot1 = int(0.7*n)
    pivot2 = int(0.8*n)
    Xtr = X[:pivot1]
    Ytr = Y[:pivot1]
    Xval = X[pivot1:pivot2]
    Yval = Y[pivot1:pivot2]
    Xtest = X[pivot2:]
    Ytest = Y[pivot2:]
    
    
    
    return Xtr, Ytr, Xval, Yval, Xtest, Ytest
