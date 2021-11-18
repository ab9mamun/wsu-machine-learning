# -*- coding: utf-8 -*-
"""

@author: Abdullah Mamun
"""

import naivebayes as nbayes
import datamanager as dm
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
    
    traindata, trainlabels, testdata, testlabels, stoplist =  dm.readAndPreprocess(dryrun=dryrun) #method from datamanager
    #still the traindata and testdata have sentences
    vocab = dm.build_vocabulary(traindata, stoplist)
    #vocab is a sorted list of all the words found in the traindata minus the stopwords
    vocab_map = dm.build_vocab_map(vocab) 
    #we created a map/dictionary for vocabulary (key = word, value = index) 
    M = len(vocab)
    
    Xtrain = dm.create_feature_vectors(traindata, vocab_map, M, stoplist)
    #now we have created the features with 1's and zeros indicating which words are present in a sentece and which words are not
    
    #Xtest = dm.create_feature_vectors(testdata, vocab_map, M, stoplist)
    '''
    There can be unseen words in the test data.
    So, there is a challenge to do the same with the test data 
    '''
    
    print(Xtrain[:2, :])
    
    '''
    naive bayes implementation starts here
    '''
    prior_y0, prior_y1 = nbayes.get_priors(trainlabels)
    
    
    
    #print(Xtest[:2, :])
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    '''
    print(traindata[0])
    print(testdata[0])
    print(trainlabels[0], type(trainlabels[0]))
    print(testlabels[0], type(testlabels[0]))
    print(stoplist[0])
    '''
    
    
   
    
    
    
    
    
if __name__ == "__main__":
    main()