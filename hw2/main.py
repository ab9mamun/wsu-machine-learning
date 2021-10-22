# -*- coding: utf-8 -*-
"""

@author: Abdullah Mamun
"""

from modelmanager import *
from datamanager import *
import numpy as np
from sklearn.utils import shuffle
import pandas as pd
'''
python version 3.7.4
'''

def main():
    np.set_printoptions(suppress=True)
    np.random.seed(0)
    dryrun = False
    reduced_data = False
    
    Xtrain, Ytrain_multi, Xtest, Ytest_multi, Ytrain_bin, Ytest_bin =  readAndPreprocessFashion(dryrun=dryrun) #method from datamanager
    if Xtrain is not None:
        print('=== Fashion Dataset loaded===')
    #the labels were already shuffled in the dataset, so we are not going to shuffle here.
    #using the simpler names for the binary labels for now

    Ytrain, Ytest = Ytrain_multi, Ytest_multi
    #Xtrain, Ytrain = shuffle(Xtrain, Ytrain, random_state=0)
    #Xtest, Ytest = shuffle(Xtest, Ytest, random_state =0)
    if reduced_data:
        Xtrain = Xtrain[:10000]
        Ytrain = Ytrain[:10000]
        Xtest = Xtest[:2000]
        Ytest = Ytest[:2000]
        

    #2.1a - SVM finding best param
    #SVM_tuning(Xtrain, Ytrain, Xtest, Ytest, "2.1a")
    #2.1b - SVM combined dataset
    #SVM_combined(Xtrain, Ytrain, Xtest, Ytest, "2.1b")
    #2.1c - SVM with polynomial kernel
    #SVM_polynomial(Xtrain, Ytrain, Xtest, Ytest, "2.1c")
    
    #2.2- Kernelized perceptron
    #kernelized_perceptron(Xtrain, Ytrain, Xtest, Ytest, "2.2")
    
    #2.3- Decision tree
    
    Xtrain, Ytrain, Xval, Yval, Xtest, Ytest = readAndPreprocessCancer(dryrun=dryrun)

        
        
    decision_tree_problem(Xtrain, Ytrain,Xval,Yval, Xtest, Ytest, "2.3")
    
   
    
    
    
    
    
if __name__ == "__main__":
    main()