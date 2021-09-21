# -*- coding: utf-8 -*-
"""
Created on Mon Sep 20 18:12:21 2021

@author: abdullahal.mamun1
"""
import pandas as pd

def read_data():
    traindf = pd.read_csv('data/fashion-mnist_train.csv')
    testdf = pd.read_csv('data/fashion-mnist_test.csv')
    return traindf, testdf


def preprocess(classifier):
    traindf, testdf = read_data()

    print(traindf.head())
    print(testdf.head())
    
    train, test = traindf.values, testdf.values
    Ytrain = train[:,0]
    Xtrain = train[:,1:]
    Ytest = test[:,0]
    Xtest = test[:, 1:]
    
    if classifier == "binary":
        Ytrain = Ytrain%2
        Ytest = Ytest%2
        Ytrain[Ytrain==0] = -1
        Ytest[Ytest==0]= -1
    
    return Xtrain, Ytrain, Xtrain, Xtest

