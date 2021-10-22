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
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
import decision_tree


def SVM_tuning(Xtrain, Ytrain, Xtest, Ytest, problemNo):
    print("Running experiments for ", problemNo)
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
    #print(Xtr.shape, Ytr.shape)
    trainaccs = []
    valaccs = []
    testaccs = []
    supports = []
    C_values = [10**(i) for i in range(-4, 5)]
    
    for C in C_values: # this array will have all possible values for C as required
        clf = make_pipeline(StandardScaler(), SVC(kernel="linear", C=C, random_state=0))
        clf.fit(Xtr, Ytr)
    
        trainacc = (np.count_nonzero(clf.predict(Xtr)==Ytr))/trlen
        valacc = (np.count_nonzero(clf.predict(Xval)==Yval))/vallen
        testacc = (np.count_nonzero(clf.predict(Xtest)==Ytest))/testlen
        support = len(clf.named_steps['svc'].support_vectors_)
        print('Accuracies for C=', C, ':', trainacc, valacc, testacc)
        print('Number of support vectors: ', support)
        trainaccs.append(trainacc)
        valaccs.append(valacc)
        testaccs.append(testacc)
        supports.append(support)
    
    plot_graph(C_values, trainaccs, 'C', 'training accuracy', 'SVM with Linear Kernel (train acc)')
    plot_graph(C_values, valaccs, 'C', 'validation accuracy', 'SVM with Linear Kernel (val acc)')
    plot_graph(C_values, testaccs, 'C', 'test accuracy', 'SVM with Linear Kernel (test acc)')
    plot_graph(C_values, supports, 'C', 'Number of support vectors', 'SVM with Linear Kernel (support vectors)')

def SVM_combined(Xtrain, Ytrain, Xtest, Ytest, problemNo):
    print("Running experiment for ", problemNo)
    clf = make_pipeline(StandardScaler(), SVC(kernel="linear", C=0.01, random_state=0))
    clf.fit(Xtrain, Ytrain)
    Ypred = clf.predict(Xtest)
    testacc = (np.count_nonzero(Ypred==Ytest))/len(Ytest)
    print("Test acc:", testacc)
    
    
    #np.set_printoptions(precision=2)
    
    # Plot non-normalized confusion matrix
    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                   'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    cm = confusion_matrix(Ytest, Ypred, labels=clf.classes_, normalize='true')
    
    
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    fig, ax = plt.subplots(figsize=(10,10))
    disp.plot(ax=ax, cmap = 'Blues')
    plt.title('Confusion Matrix (Linear SVM, trained on combined data)')
    try:
        plt.savefig('Confusion Matrix (Linear SVM, trained on combined data).png')
    except:
        pass
    plt.show()
    
    

def SVM_polynomial(Xtrain, Ytrain, Xtest, Ytest, problemNo):
    print("Running experiments for ", problemNo)
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
    #print(Xtr.shape, Ytr.shape)
    trainaccs = []
    valaccs = []
    testaccs = []
    supports = []
    degree_values = [2, 3, 4]
    
    for degree in degree_values: # this array will have all possible values for C as required
        clf = make_pipeline(StandardScaler(), SVC(kernel="poly", C=0.01, random_state=0))
        clf.fit(Xtr, Ytr)
    
        trainacc = (np.count_nonzero(clf.predict(Xtr)==Ytr))/trlen
        valacc = (np.count_nonzero(clf.predict(Xval)==Yval))/vallen
        testacc = (np.count_nonzero(clf.predict(Xtest)==Ytest))/testlen
        support = len(clf.named_steps['svc'].support_vectors_)
        print('Accuracies for C=', C, ':', trainacc, valacc, testacc)
        print('Number of support vectors: ', support)
        trainaccs.append(trainacc)
        valaccs.append(valacc)
        testaccs.append(testacc)
        supports.append(support)
    
    plot_graph(C_values, trainaccs, 'C', 'training accuracy', 'SVM with Linear Kernel (train acc)')
    plot_graph(C_values, valaccs, 'C', 'validation accuracy', 'SVM with Linear Kernel (val acc)')
    plot_graph(C_values, testaccs, 'C', 'test accuracy', 'SVM with Linear Kernel (test acc)')
    plot_graph(C_values, supports, 'C', 'Number of support vectors', 'SVM with Linear Kernel (support vectors)')


def kernelized_perceptron(Xtrain, Ytrain, Xtest, Ytest, problemNo):
    pass

def decision_tree_problem(Xtrain, Ytrain,Xval,Yval, Xtest, Ytest, problemNo):
    decision_tree.run(Xtrain, Ytrain, Xval,Yval, Xtest, Ytest, problemNo)


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
    
        