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
    #shuffle()
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

def online_learning_multiclass(Xtrain, Ytrain, Xtest, Ytest, epochs, algorithm):
    
    n,d = Xtrain.shape #n = number of examples, d = size of the feature vector
    k = 10
    w = np.random.uniform(-0.5, 0.5, (k,d))
    #w = np.zeros((d,))
    accuracies = []
    mistakes = []
    prods = [0]*k
    #shuffle()
    for iteration in range(epochs):
        correct = 0
        for t in range(n):            
            x, y = Xtrain[t], Ytrain[t]
            #Calculating yhat with the argmax function (Algorithm 2, line 4)
            #F is a sparse matrix where every row is zero except the row for the correct class which contains the feature vector
            for y_var in range(k):
                  #no need to declare a matrix. saving memory and time
                #F is a sparse matrix where every row is zero except the row for the correct class which contains the feature vector
                '''
                F = np.zeros((k,d))
                F[y_var] = x
                prods[y_var] = np.sum(np.dot(w, np.transpose(F))) #no need to execute this expensive line so many times
                '''
                prods[y_var] = np.sum(np.dot(w[y_var],x))  #same calculation but different formula just to save some computation time
                                                #just taking the advantage of sparsity
            yhat = np.argmax(prods)
            #End of yhat calculation
            
            if yhat != y: #if mistake then
                '''
                Fhat = np.zeros((k,d))
                Fhat[yhat] = x
                F = np.zeros((k,d))
                F[y] = x
                Fdiff = F - Fhat
                yprod = np.sum(np.dot(w,np.transpose(F)))
                yhatprod = np.sum(np.dot(w,np.transpose(Fhat)))
                '''
                #saving time and memory by taking advantage of sparsity
                
                yprod = np.sum(np.dot(w[y],x))
                yhatprod = np.sum(np.dot(w[y], x))
                
                if t % (n//20) == 0:
                    print('-',end='')
                if algorithm == "standard":
                    '''
                    w = w + Fdiff
                    '''
                    w[y]= w[y]+ x
                    w[yhat] = w[yhat]- x #equivalent but cheaper operation
                elif algorithm == "pa":
                    '''
                    tau = (1 - (yprod - yhatprod))/ np.sum(Fdiff*Fdiff)   ##the denominator is the square of the Frobenious norm
                    w = w + tau*Fdiff
                    '''
                    #optimized for sparsity
                    tau = (1 - (yprod - yhatprod))/ (2*np.dot(x,x))     #Fdiff has a row with x and another with -x. So, the frobenious norm will be twice the L2 norm of x
                    w[y] = w[y] + tau*x
                    w[yhat]= w[yhat] - tau*x
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




def standard_binary(Xtrain, Ytrain, Xtest, Ytest, epochs, quesNo):
    print("Running the standard_binary for Q", quesNo)
    w = online_learning_binary(Xtrain, Ytrain, Xtest, Ytest, epochs, "standard")


def standard_multiclass(Xtrain, Ytrain, Xtest, Ytest, epochs, quesNo):
    print("Running the standard_multiclass for Q", quesNo)
    w = online_learning_multiclass(Xtrain, Ytrain, Xtest, Ytest, epochs, "standard")


def pa_binary(Xtrain, Ytrain, Xtest, Ytest, epochs, quesNo):
    print("Running the pa_binary for Q", quesNo)
    w = online_learning_binary(Xtrain, Ytrain, Xtest, Ytest, epochs, "pa")

def pa_multiclass(Xtrain, Ytrain, Xtest, Ytest, epochs, quesNo):
    print("Running the pa_multiclass for Q", quesNo)
    w = online_learning_multiclass(Xtrain, Ytrain, Xtest, Ytest, epochs, "pa")

def averaged_binary(Xtrain, Ytrain, Xtest, Ytest, epochs, quesNo):
    print("Running the averaged_binary for Q", quesNo)

def averaged_multiclass(Xtrain, Ytrain, Xtest, Ytest, epochs, quesNo):
    print("Running the averaged_multiclass for Q", quesNo)
    

def general_binary(Xtrain, Ytrain, Xtest, Ytest, epochs, quesNo, dryrun):
    print("Running the general_binary for Q", quesNo)


def general_multiclass(Xtrain, Ytrain, Xtest, Ytest, epochs, quesNo, dryrun):
    print("Running the general_multiclass for Q", quesNo)
        