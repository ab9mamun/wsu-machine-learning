# -*- coding: utf-8 -*-
"""
Created on Mon Sep 20 18:01:54 2021

@author: abdullahal.mamun1
"""

from modelmanager import *

import pandas as pd
from datamanager import read_data

def main():
    dryrun = True
    print('hello')
    #traindf, testdf = read_data()
    print('Hellos')
    
    Xtrain, Ytrain, Xtest, Ytest = None,None,None,None

    
    #5.1a - standard and binary
    standard_binary(Xtrain, Ytrain, Xtest, Ytest, 50, "5.1a")
    pa_binary(Xtrain, Ytrain, Xtest, Ytest, 50, "5.1a")
    
    #5.1b,c
    standard_binary(Xtrain, Ytrain, Xtest, Ytest, 20, "5.1b")
    pa_binary(Xtrain, Ytrain, Xtest, Ytest, 20, "5.1b")
    
    averaged_binary(Xtrain, Ytrain, Xtest, Ytest, 20, "5.1c")
    
    #5.1d 
    general_binary(Xtrain, Ytrain, Xtest, Ytest, 20, "5.1d", dryrun)
    
    
    #5.1a - standard and binary
    standard_binary(Xtrain, Ytrain, Xtest, Ytest, 50, "5.1a")
    pa_binary(Xtrain, Ytrain, Xtest, Ytest, 50, "5.1a")
    
    #5.1b,c
    standard_binary(Xtrain, Ytrain, Xtest, Ytest, 20, "5.1b")
    pa_binary(Xtrain, Ytrain, Xtest, Ytest, 20, "5.1b")
    
    averaged_binary(Xtrain, Ytrain, Xtest, Ytest, 20, "5.1c")
    
    #5.1d 
    general_binary(Xtrain, Ytrain, Xtest, Ytest, 20, "5.1d", dryrun)
    
    
    
    
    
if __name__ == "__main__":
    main()