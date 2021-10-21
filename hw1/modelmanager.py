# -*- coding: utf-8 -*-
"""
@author: Abdullah Mamun
"""
import numpy as np
from sklearn.utils import shuffle
from copy import deepcopy
import matplotlib.pyplot as plt

def online_learning_binary(Xtrain, Ytrain, Xtest, Ytest, epochs, algorithm, quesNo):
    
    n,d = Xtrain.shape #n = number of examples, d = size of the feature vector
    w = np.random.uniform(-0.5, 0.5, (d,))
    #w = np.zeros((d,))
    train_accuracies = []
    train_mistakes = []
    test_accuracies = []
    #shuffle()
    for iteration in range(epochs):
        correct = 0
        for t in range(n):
            
            x, y = Xtrain[t], Ytrain[t]
            yhat = np.sign(np.dot(w, x))  #prediction is made
            
            if yhat != y: #if mistake then

                #if t % (n//20) == 0:
                #    print('-',end='')
                if algorithm == "standard":
                    w = w + 1*y*x
                elif algorithm == "pa":
                    tau = (1-np.dot(y,np.dot(w,x)))/np.dot(x,x)  #np.dot(x,x)= || x ||^2; we could also write np.linalg.norm(x)**2
                    w = w + tau*y*x
                else:
                    print("ERROR: Algorithm", algorithm, "is not defined.")
            else:
                correct+=1
                #if t % (n//20) == 0:
                #    print('+',end='')
        
        train_acc = correct/n
        
        if (quesNo == "5.1b" or iteration == epochs-1) : #need to calculate test accuracies
            test_corr = 0
            #print('/', end='')
            for t in range(Xtest.shape[0]):
                x, y = Xtest[t], Ytest[t]
                yhat = np.sign(np.dot(w,x))
                if yhat != y:
                    pass
                    #if t % (n//10) == 0:
                    #    print('-',end='')
                else:
                    #if t % (n//10) == 0:
                    #    print('+',end='')
                    test_corr+=1
            test_acc = test_corr/Xtest.shape[0]
            if quesNo != "5.1d" and (iteration+1)% (epochs//5) == 0:
                print("Training iter:{}/{}, train_acc={}, test_acc={}".format(iteration+1, epochs,round(train_acc,4), round(test_acc,4)))
            test_accuracies.append(test_acc)
            
        elif quesNo in  ["5.1a"] and (iteration+1)% (epochs//5) == 0:
            # no need to calculate test accuracies
            print("Training iter:{}/{}, train_acc={}".format(iteration+1, epochs,round(train_acc,4)))
            
        train_accuracies.append(train_acc)
        train_mistakes.append(n-correct)
    
    if quesNo != "5.1d":
        print("Final test accuracy: ",test_accuracies[-1])
    
    # time to plot the graphs
    epochlist = list(range(1,epochs+1))
    if quesNo == "5.1a":
        plot_graph(epochlist,train_mistakes,'iteration', 'training mistakes', 'Q {}- Training mistakes for {} BINARY perceptron'.format(quesNo, algorithm.upper()))
    elif quesNo == "5.1b":
        plot_graph(epochlist,train_accuracies,'iteration', 'training accuracy', 'Q {}- Training accuracies for {} BINARY perceptron'.format(quesNo, algorithm.upper()))
        plot_graph(epochlist,test_accuracies,'iteration', 'test accuracy', 'Q {}- Test accuracies for {} BINARY perceptron'.format(quesNo, algorithm.upper()))
    
    return w, test_accuracies[-1]

def online_learning_multiclass(Xtrain, Ytrain, Xtest, Ytest, epochs, algorithm, quesNo):
    
    n,d = Xtrain.shape #n = number of examples, d = size of the feature vector
    k = 10
    w = np.random.uniform(-0.5, 0.5, (k,d))
    #w = np.zeros((d,))
    train_accuracies = []
    train_mistakes = []
    test_accuracies = []
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
                
                #if t % (n//20) == 0:
                #    print('-',end='')
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
                #if t % (n//20) == 0:
                #    print('+',end='')
        
        train_acc = correct/n
            
        if quesNo == "5.2b" or iteration == epochs-1 : #need to calculate test accuracies
            test_corr = 0
            #print('/',end='')
            for t in range(Xtest.shape[0]):
                x, y = Xtest[t], Ytest[t]
                for y_var in range(k):
                    prods[y_var] = np.sum(np.dot(w[y_var], x))
                yhat = np.argmax(prods)
                
                if yhat != y:
                    pass
                    #if t % (n//10) == 0:
                    #    print('-',end='')
                else:
                    #if t % (n//10) == 0:
                    #    print('+',end='')
                    test_corr+=1
                    
            test_acc = test_corr/Xtest.shape[0]
            if quesNo != "5.2d" and (iteration+1)% (epochs//5) == 0:
                print("Training iter:{}/{}, train_acc={}, test_acc={}".format(iteration+1, epochs,round(train_acc,4), round(test_acc,4)))
            test_accuracies.append(test_acc)
            
        elif quesNo in  ["5.2a"] and (iteration+1)% (epochs//5) == 0:
            # no need to calculate test accuracies
            print("Training iter:{}/{}, train_acc={}".format(iteration+1, epochs,round(train_acc,4)))
            
        train_accuracies.append(train_acc)
        train_mistakes.append(n-correct)
    if quesNo != "5.2d":
        print("Final test accuracy: ",test_accuracies[-1])
        
    epochlist = list(range(1,epochs+1))
    if quesNo == "5.2a":
        plot_graph(epochlist,train_mistakes,'iteration', 'training mistakes', 'Q {}- Training mistakes for {} MULTICLASS perceptron'.format(quesNo, algorithm.upper()))
    elif quesNo == "5.2b":
        plot_graph(epochlist,train_accuracies,'iteration', 'training accuracy', 'Q {}- Training accuracies for {} MULTICLASS perceptron'.format(quesNo, algorithm.upper()))
        plot_graph(epochlist,test_accuracies,'iteration', 'test accuracy', 'Q {}- Test accuracies for {} MULTICLASS perceptron'.format(quesNo, algorithm.upper()))
    return w, test_accuracies[-1]



'''
Adapted from Hal's Algorithm 7- AveragedPerceptronTrain
Modifications: * ignored the bias term
'''
def online_learning_averaged_binary(Xtrain, Ytrain, Xtest, Ytest, epochs):
    
    n,d = Xtrain.shape #n = number of examples, d = size of the feature vector
    w = np.random.uniform(-0.5, 0.5, (d,))
    u = deepcopy(w)
    c = 1
    #w = np.zeros((d,))
    train_accuracies = []
    test_accuracies = []
    #mistakes = []
    #shuffle()
    for iteration in range(epochs):
        correct = 0
        for t in range(n):
            
            x, y = Xtrain[t], Ytrain[t]
            yhat = np.sign(np.dot(w, x))  #prediction is made
            
            if yhat != y: #if mistake then
                #if t % (n//20) == 0:
                #    print('-',end='')
                w = w + y*x
                u = u + y*c*x
            else:
                correct+=1
                #if t % (n//20) == 0:
                #    print('+',end='')
            c+= 1
            '''
            this is what was suggested in the algorithm,
            we increment the cunter regardless of the update
            '''
                
      
        #print("/",end='')   
        train_acc = correct/n
        w_avg = w - u/c
        test_corr = 0
        for t in range(Xtest.shape[0]):
            x, y = Xtest[t], Ytest[t]
            yhat = np.sign(np.dot(w_avg,x))
            if yhat != y:
                pass
                #if t% (n//10) ==0:
                #    print('-', end='')
            else:
                #if t% (n//10) ==0:
                #    print('+', end='')
                test_corr+=1
        
                    
        #print()
        test_acc = test_corr/Xtest.shape[0]
        if (iteration+1)% (epochs//5) == 0:
            print("Training iter:{}/{}, train_acc={}, test_acc={}".format(iteration+1, epochs,round(train_acc,4), round(test_acc,4)))
        test_accuracies.append(test_acc)
        train_accuracies.append(train_acc)
        #test_
        #mistakes.append(n-correct)
     
    epochlist = list(range(1,epochs+1))
    print("Final test accuracy: ",test_accuracies[-1])
    plot_graph(epochlist,train_accuracies,'iteration', 'training accuracy', 'Q {}- Training accuracies for {} BINARY perceptron'.format("5.1c", "averaged".upper()))
    plot_graph(epochlist,test_accuracies,'iteration', 'test accuracy', 'Q {}- Test accuracies for {} BINARY perceptron'.format("5.1c", "averaged".upper()))
    return w-u/c, test_accuracies[-1]

'''
Adapted from Hal's Algorithm 7- AveragedPerceptronTrain
Modifications: * ignored the bias term
               * adapted for multiclass
'''
def online_learning_averaged_multiclass(Xtrain, Ytrain, Xtest, Ytest, epochs):
    k = 10
    n,d = Xtrain.shape #n = number of examples, d = size of the feature vector
    w = np.random.uniform(-0.5, 0.5, (k,d))
    u = deepcopy(w)
    c = 1
    #w = np.zeros((d,))
    train_accuracies = []
    test_accuracies = []
    #mistakes = []
    #shuffle()
    prods = [0]*k
        
    for iteration in range(epochs):
        correct = 0
        for t in range(n):
            
            x, y = Xtrain[t], Ytrain[t]
            for y_var in range(k):
                prods[y_var] = np.sum(np.dot(w[y_var],x))  #same calculation but different formula just to save some computation time
                '''see the comments of the online_learning_multiclass function for clarification'''
            yhat = np.argmax(prods) #prediction is made
            
            if yhat != y: #if mistake then
                #if t % (n//20) == 0:
                #   print('-',end='')
                w[y] = w[y] + x
                w[yhat] = w[yhat] - x
                u[y] = u[y] + c*x
                u[yhat] = u[yhat] - c*x
                '''
                Again, the multiclass algorithm was optimized a little bit
                by replacing the expensive statements with equivalent cheaper ones.
                See the comments of online_learning_multiclass function for clarification.
                '''
            else:
                correct+=1
                #if t % (n//20) == 0:
                #    print('+',end='')
            c+= 1
                
        train_acc = correct/n  ##this is the training accuracy
        #print("/",end='')   
        w_avg = w - u/c
        test_corr = 0
        for t in range(Xtest.shape[0]):
            x, y = Xtest[t], Ytest[t]
            for y_var in range(k):
                prods[y_var] = np.sum(np.dot(w[y_var],x))  #same calculation but different formula just to save some computation time
                '''see the comments of the online_learning_multiclass function for clarification'''
            yhat = np.argmax(prods)
            if yhat != y:
                pass
                #if t% (n//10) ==0:
                #    print('-', end='')
            else:
                #if t% (n//10) ==0:
                #   print('+', end='')
                test_corr+=1
        
                    
        #print()
        test_acc = test_corr/Xtest.shape[0]  ##this is the test accuracy after this iteration
        if (iteration+1)% (epochs//5) == 0:
            print("Training iter:{}/{}, train_acc={}, test_acc={}".format(iteration+1, epochs,round(train_acc,4), round(test_acc,4)))
        test_accuracies.append(test_acc)
        train_accuracies.append(train_acc)
        #test_
        #mistakes.append(n-correct)
        
    epochlist = list(range(1,epochs+1))
    print("Final test accuracy: ",test_accuracies[-1])
    plot_graph(epochlist,train_accuracies,'iteration', 'training accuracy', 'Q {}- Training accuracies for {} MULTICLASS perceptron'.format("5.2c", "averaged".upper()))
    plot_graph(epochlist,test_accuracies,'iteration', 'test accuracy', 'Q {}- Test accuracies for {} MULTICLASS perceptron'.format("5.2c", "averaged".upper()))
    return w-u/c, test_accuracies[-1]

def standard_binary(Xtrain, Ytrain, Xtest, Ytest, epochs, quesNo):
    print("==================================================================")
    print("Q ",quesNo,"\n-------------")
    print("Running the standard_binary for Q", quesNo)
    w, test_acc = online_learning_binary(Xtrain, Ytrain, Xtest, Ytest, epochs, "standard", quesNo)


def standard_multiclass(Xtrain, Ytrain, Xtest, Ytest, epochs, quesNo):
    print("==================================================================")
    print("Q ",quesNo,"\n-------------")
    print("Running the standard_multiclass for Q", quesNo)
    w, test_acc = online_learning_multiclass(Xtrain, Ytrain, Xtest, Ytest, epochs, "standard", quesNo)


def pa_binary(Xtrain, Ytrain, Xtest, Ytest, epochs, quesNo):
    print("==================================================================")
    print("Q ",quesNo,"\n-------------")
    print("Running the pa_binary for Q", quesNo)
    w, test_acc = online_learning_binary(Xtrain, Ytrain, Xtest, Ytest, epochs, "pa", quesNo)

def pa_multiclass(Xtrain, Ytrain, Xtest, Ytest, epochs, quesNo):
    print("==================================================================")
    print("Q ",quesNo,"\n-------------")
    print("Running the pa_multiclass for Q", quesNo)
    w, test_acc = online_learning_multiclass(Xtrain, Ytrain, Xtest, Ytest, epochs, "pa", quesNo)

def averaged_binary(Xtrain, Ytrain, Xtest, Ytest, epochs, quesNo):
    print("==================================================================")
    print("Q ",quesNo,"\n-------------")
    print("Running the averaged_binary for Q", quesNo)
    w, test_acc = online_learning_averaged_binary(Xtrain, Ytrain, Xtest, Ytest, epochs)

def averaged_multiclass(Xtrain, Ytrain, Xtest, Ytest, epochs, quesNo):
    print("==================================================================")
    print("Q ",quesNo,"\n-------------")
    print("Running the averaged_multiclass for Q", quesNo)
    w, test_acc = online_learning_averaged_multiclass(Xtrain, Ytrain, Xtest, Ytest, epochs)


def general_binary(Xtrain, Ytrain, Xtest, Ytest, epochs, quesNo, dryrun):
    print("==================================================================")
    print("Q ",quesNo,"\n-------------")
    print("Running the general_binary for Q", quesNo)
    training_example_counts = [(i+1)*100 for i in range(600)]
    test_accuracies = []
    print("Printing (#train_examples,test_acc):",end='')
    for t_idx in range(600):
        
        Xtrain_temp = Xtrain[:training_example_counts[t_idx], :]
        Ytrain_temp = Ytrain[:training_example_counts[t_idx]]
        w, test_acc = online_learning_binary(Xtrain_temp, Ytrain_temp, Xtest, Ytest, epochs, "standard", quesNo)
        test_accuracies.append(test_acc)
        print("({},{})".format(training_example_counts[t_idx], round(test_acc,4)),end="; ")
    
    print()
    plot_graph(training_example_counts,test_accuracies,'# training examples', 'test accuracy', 'Q {}- General learning curve for STANDARD BINARY perceptron'.format(quesNo))
    
    


def general_multiclass(Xtrain, Ytrain, Xtest, Ytest, epochs, quesNo, dryrun):
    print("==================================================================")
    print("Q ",quesNo,"\n-------------")
    print("Running the general_multiclass for Q", quesNo)
    training_example_counts = [(i+1)*100 for i in range(20)]
    test_accuracies = []
    print("Printing (#train_examples,test_acc):",end='')
    for t_idx in range(20):
        
        Xtrain_temp = Xtrain[:training_example_counts[t_idx], :]
        Ytrain_temp = Ytrain[:training_example_counts[t_idx]]
        w, test_acc = online_learning_multiclass(Xtrain_temp, Ytrain_temp, Xtest, Ytest, epochs, "standard", quesNo)
        test_accuracies.append(test_acc)
        print("({},{})".format(training_example_counts[t_idx], round(test_acc,4)),end="; ")
    
    print()
    plot_graph(training_example_counts,test_accuracies,'# training examples', 'test accuracy', 'Q {}- General learning curve for STANDARD MULTICLASS perceptron'.format(quesNo))


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
    
        