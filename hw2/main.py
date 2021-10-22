# -*- coding: utf-8 -*-
"""

@author: Abdullah Mamun
"""

from modelmanager import *
from datamanager import *
import numpy as np
from sklearn.utils import shuffle
import pandas as pd
from datamanager import read_data
'''
python version 3.7.4
'''

def main():
    np.set_printoptions(suppress=True)
    np.random.seed(0)
    dryrun = False
    
    Xtrain, Ytrain_multi, Xtest, Ytest_multi, Ytrain_bin, Ytest_bin =  readAndPreprocess(dryrun=dryrun) #method from datamanager
    if Xtrain is not None:
        print('===Dataset loaded===')
    #the labels were already shuffled in the dataset, so we are not going to shuffle here.
    #using the simpler names for the binary labels for now

    Ytrain, Ytest = Ytrain_multi, Ytest_multi
    #Xtrain, Ytrain = shuffle(Xtrain, Ytrain, random_state=0)
    #Xtest, Ytest = shuffle(Xtest, Ytest, random_state =0)
    dryrun = True
    if dryrun:
        Xtrain = Xtrain[:200]
        Ytrain = Ytrain[:200]
        Xtest = Xtest[:100]
        Ytest = Ytest[:100]
        

    #2.1a - SVM finding best param
    SVM_tuning(Xtrain, Ytrain, Xtest, Ytest, "2.1a")
   
    
    
    
    
    
if __name__ == "__main__":
    main()