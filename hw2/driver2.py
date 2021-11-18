# -*- coding: utf-8 -*-
"""
@author: Abdullah Mamun
"""
import pandas
import decision_tree2
data = pandas.read_csv('data/breast-cancer-wisconsin.data', header=None)

data.replace('?', 4, inplace=True)  
data = data.values
X = data[:,1:10]
Y = data[:,10]  
Y[Y==2] = 0
Y[Y==4] = 1 
X = X.astype(float)
Y = Y.astype(int)

n = len(X)
trainX = X[:int(0.7*n)]
trainY = Y[:int(0.7*n)]
valX = X[int(0.7*n):int(0.8*n)]
valY = Y[int(0.7*n):int(0.8*n)]
testX = X[int(0.8*n):]
testY = Y[int(0.8*n):]

decision_tree2.createAndTest(trainX, valX, testX, trainY, valY, testY)
    