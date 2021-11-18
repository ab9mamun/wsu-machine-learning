# -*- coding: utf-8 -*-
"""

@author: Abdullah Mamun
"""

import naivebayes as nbayes
import datamanager as dm
import numpy as np
from sklearn.utils import shuffle
import sys
import pandas as pd
'''
python version 3.7.4
'''


def main():
    np.set_printoptions(suppress=True)
    np.random.seed(0)
    dryrun = False
    
    '''
    ============================================================
    Redirecting the output to the file............
    =============================================================
    '''
    stdout_bak = sys.stdout
    stderr_bak = sys.stderr
    file = open('output.txt', 'w')
    sys.stdout = file
    sys.stderr = file
    
    '''
    ============================================================
    Building the vocabulary and creating the training features
    =============================================================
    '''
    traindata, trainlabels, testdata, testlabels, stoplist =  dm.readAndPreprocess(dryrun=dryrun) #method from datamanager
    #still the traindata and testdata have sentences
    vocab = dm.build_vocabulary(traindata, stoplist)
    #vocab is a sorted list of all the words found in the traindata minus the stopwords
    vocab_map = dm.build_vocab_map(vocab) 
    #we created a map/dictionary for vocabulary (key = word, value = index) 
    M = len(vocab)
    n = len(traindata)
    
    Xtrain = dm.create_feature_vectors(traindata, vocab_map, M, stoplist)
    #Xtrain, trainlabels, traindata = shuffle(Xtrain, trainlabels, traindata)
    #now we have created the features with 1's and zeros indicating which words are present in a sentece and which words are not
    
    #Xtest = dm.create_feature_vectors(testdata, vocab_map, M, stoplist)
    '''
    There can be unseen words in the test data.
    So, there is a challenge to do the same with the test data 
    '''
    
    #print(Xtrain[:2, :])
    
    '''
    naive bayes implementation starts here
    '''
    '''
    ============================================================
    Calculating the priors and the likelihoods............
    =============================================================
    '''
    print("Naive Bayes implementation starts")
    prior_y0, prior_y1 = nbayes.get_priors(trainlabels)
    print("The prior beliefs---\nP(y=0) =",prior_y0, "P(y=1) =", prior_y1)
    x_given_y0, x_given_y1 = nbayes.get_x_given_y(Xtrain, trainlabels, M)
    
    print("Calculated P(x | y=0) for all words in the vocabulary")
    #print(x_given_y0)
    print("Calculated P(x | y=1) for all words in the vocabulary")
    #print(x_given_y1)
    
    '''
    ============================================================
    Predicting on the training data............
    =============================================================
    '''
    
    y_train_pred = nbayes.predict_on_train(Xtrain, prior_y0, prior_y1, x_given_y0, x_given_y1)
    print("Calculated Training predictions: ")
    #print(y_train_pred)
    correct = np.count_nonzero(trainlabels==y_train_pred)
    trainAcc = correct/n
    print("Training accuracy:", trainAcc)
    
    '''
    ============================================================
    Predicting on the test data............
    ============================================================
    '''
    ntest = len(testdata)
    Xtest = dm.create_feature_vectors(testdata, vocab_map, M, stoplist)
    '''
    Shuffling doesn't have any effect on training or testing
    on NaiveBayes. So, we are not shuffling to keep the reference indices same
    with the original dataset for comparison
    '''
    #Xtest, testlabels, testdata = shuffle(Xtest, testlabels, testdata)
   
    y_test_pred = nbayes.predict_on_test(Xtest, prior_y0, prior_y1, x_given_y0, x_given_y1, n)
    print("Calculated Test predictions: ")
    #print(y_test_pred)
    correct = np.count_nonzero(testlabels==y_test_pred)
    testAcc = correct/ntest
    print("Test accuracy:", testAcc)
    
    
    '''
    ============================================================
    Validating with scikit-learn.....................
    =============================================================
    '''
    
    print("\nValidating with scikit-learn")
    from sklearn.naive_bayes import MultinomialNB
    clf = MultinomialNB()
    clf.fit(Xtrain, trainlabels)
    ytrainpred_scikit = clf.predict(Xtrain)
    ytestpred_scikit = clf.predict(Xtest)
    '''
    ============================================================
    scikit predictions are ready. Now time to calculate accuracies
    =============================================================
    '''
    
    correct = np.count_nonzero(trainlabels == ytrainpred_scikit)
    print("Train Acc of scikit:", correct/n)
    correct = np.count_nonzero(testlabels == ytestpred_scikit)
    print("Test Acc of scikit:", correct/ntest)
    
    '''
    ============================================================
    Checking on how many examples, our NaiveBayes
    and scikit disagree................
    =============================================================
    '''
    disagreements = n- np.count_nonzero(ytrainpred_scikit == y_train_pred)
    print("Our NaiveBayes and scikit disagree on",disagreements, "out of", n, "training examples")
    disagreements = ntest -np.count_nonzero(ytestpred_scikit == y_test_pred)
    print("Our NaiveBayes and scikit disagree on",disagreements, "out of", ntest, "test examples")
    
    print("\nYou can also check trainpredictions.csv and testpredictions.csv to see the individual predictions")
    
    
    '''
    ============================================================
    Resetting the output stream............
    =============================================================
    '''
    sys.stdout = stdout_bak
    sys.stderr = stderr_bak
    file.close()
    print("Check output.txt for the output of the program")
    print("You can also check trainpredictions.csv and testpredictions.csv to see the individual predictions")
  
    '''
    ============================================================
    Creating the CSV files......................
    =============================================================
    '''
    traincsv = pd.DataFrame()
    testcsv = pd.DataFrame()
    traincsv['Real trainlabels'] = trainlabels
    traincsv['Our predictions'] = y_train_pred
    traincsv['scikit predictions'] = ytrainpred_scikit
    
    testcsv['Real testlabels'] = testlabels
    testcsv['Our predictions'] = y_test_pred
    testcsv['scikit predictions'] = ytestpred_scikit
    
    traincsv.to_csv('trainpredictions.csv')
    testcsv.to_csv('testpredictions.csv')
    #print(traincsv.head())
    #print(testcsv.head())
    
if __name__ == "__main__":
    main()