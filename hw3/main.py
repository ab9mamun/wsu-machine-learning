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
    
    traindata, trainlabels, testdata, testlabels, stoplist =  readAndPreprocess(dryrun=dryrun) #method from datamanager
    vocab = build_vocabulary(traindata, stoplist)
    vocab_map = build_vocab_map(vocab) #create a map/dictionary for vocabulary (key = word, value = index)
    
    Xtrain, Ytrain = create_feature_vectors(vocab_map)
    #the vocabulary is ready
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    '''
    print(traindata[0])
    print(testdata[0])
    print(trainlabels[0], type(trainlabels[0]))
    print(testlabels[0], type(testlabels[0]))
    print(stoplist[0])
    '''
    
    
   
    
    
    
    
    
if __name__ == "__main__":
    main()