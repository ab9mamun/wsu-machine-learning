# -*- coding: utf-8 -*-
"""
@author: Abdullah Mamun
"""
import numpy as np
from sklearn.utils import shuffle

def online_learning_binary(Xtrain, Ytrain, Xtest, Ytest, epochs, algorithm):
    
    n,d = Xtrain.shape #n = number of examples, d = size of the feature vector
    w = np.random.uniform(-0.5, 0.5, (d,))
    #w = np.zeros((d,))
    accuracies = []
    mistakes = []
    shuffle()
    for iteration in range(epochs):
        correct = 0
        for t in range(n):
            
            x, y = Xtrain[t], Ytrain[t]
            yhat = np.sign(np.dot(w, x))  #prediction is made
            
            if yhat != y: #if mistake then
                if t % (n//20) == 0:
                    print('-',end='')
                if algorithm == "standard":
                    w = w + 1*y*x
                elif algorithm == "pa":
                    tau = (1-np.dot(y,np.dot(w,x)))/np.dot(x,x)  #np.dot(x,x)= || x ||^2; we could also write np.linalg.norm(x)**2
                    w = w + tau*y*x
                else:
                    print("ERROR: Algorithm", algorithm, "is not defined.")
            else:
                correct+=1
                if t % (n//20) == 0:
                    print('+',end='')
            
        print()   
        acc = correct/n*100
        print("Training iter:{}/{}, acc={}%".format(iteration+1, epochs,round(acc,2)))
        accuracies.append(acc)
        mistakes.append(n-correct)
        
    print("Accuracies over iterations: ",accuracies)
    return w

def online_learning_multiclass():
    pass



def standard_binary(Xtrain, Ytrain, Xtest, Ytest, epochs, quesNo):
    print("Running the standard_binary for Q", quesNo)
    w = online_learning_binary(Xtrain, Ytrain, Xtest, Ytest, epochs, "standard")


def standard_multiclass(Xtrain, Ytrain, Xtest, Ytest, epochs, quesNo):
    print("Running the standard_multiclass for Q", quesNo)


def pa_binary(Xtrain, Ytrain, Xtest, Ytest, epochs, quesNo):
    print("Running the pa_binary for Q", quesNo)
    w = online_learning_binary(Xtrain, Ytrain, Xtest, Ytest, epochs, "pa")

def pa_multiclass(Xtrain, Ytrain, Xtest, Ytest, epochs, quesNo):
    print("Running the pa_multiclass for Q", quesNo)


def averaged_binary(Xtrain, Ytrain, Xtest, Ytest, epochs, quesNo):
    print("Running the averaged_binary for Q", quesNo)

def averaged_multiclass(Xtrain, Ytrain, Xtest, Ytest, epochs, quesNo):
    print("Running the averaged_multiclass for Q", quesNo)
    

def general_binary(Xtrain, Ytrain, Xtest, Ytest, epochs, quesNo, dryrun):
    print("Running the general_binary for Q", quesNo)


def general_multiclass(Xtrain, Ytrain, Xtest, Ytest, epochs, quesNo, dryrun):
    print("Running the general_multiclass for Q", quesNo)
        