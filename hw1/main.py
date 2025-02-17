# -*- coding: utf-8 -*-
"""

@author: Abdullah Mamun
"""

from modelmanager import *
from datamanager import *
import numpy as np

import pandas as pd
from datamanager import read_data
'''
python version 3.7.4
'''

def main():
    np.set_printoptions(suppress=True)
    dryrun = False
    
    Xtrain, Ytrain_multi, Xtest, Ytest_multi, Ytrain_bin, Ytest_bin =  readAndPreprocess(dryrun=dryrun) #method from datamanager
    if Xtrain is not None:
        print('===Dataset loaded===')
    #the labels were already shuffled in the dataset, so we are not going to shuffle here.
    #using the simpler names for the binary labels for now
    Ytrain = Ytrain_bin
    Ytest = Ytest_bin
    #return
    #5.1a - standard and PA for binary classifier
    standard_binary(Xtrain, Ytrain, Xtest, Ytest, 50, "5.1a")
    pa_binary(Xtrain, Ytrain, Xtest, Ytest, 50, "5.1a")
    #return
    #5.1b,c
    standard_binary(Xtrain, Ytrain, Xtest, Ytest, 20, "5.1b")
    pa_binary(Xtrain, Ytrain, Xtest, Ytest, 20, "5.1b")
    
    averaged_binary(Xtrain, Ytrain, Xtest, Ytest, 20, "5.1c")
    
    #5.1d 
    general_binary(Xtrain, Ytrain, Xtest, Ytest, 20, "5.1d", dryrun)
    
    
    #using the simpler names for the multiclass labels for now  
    Ytrain = Ytrain_multi
    Ytest = Ytest_multi
    
    #5.2a - standard and PA for multiclass classifier
    standard_multiclass(Xtrain, Ytrain, Xtest, Ytest, 50, "5.2a")
    #return
    pa_multiclass(Xtrain, Ytrain, Xtest, Ytest, 50, "5.2a")
    
    #5.2b,c
    standard_multiclass(Xtrain, Ytrain, Xtest, Ytest, 20, "5.2b")
    pa_multiclass(Xtrain, Ytrain, Xtest, Ytest, 20, "5.2b")
    
    averaged_multiclass(Xtrain, Ytrain, Xtest, Ytest, 20, "5.2c")
    
    #5.d 
    general_multiclass(Xtrain, Ytrain, Xtest, Ytest, 20, "5.2d", dryrun)
    
    
    
    
    
if __name__ == "__main__":
    main()