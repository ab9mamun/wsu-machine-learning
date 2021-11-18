# -*- coding: utf-8 -*-
"""
@author: Abdullah Mamun
"""
import pandas as pd
import numpy as np

def fileToVector(filename):
    with open('data/'+filename) as f:
        vector = f.read().strip().split('\n')
    return vector

def read_data():    
    traindata = fileToVector('traindata.txt')
    trainlabels = fileToVector('trainlabels.txt')
    testdata = fileToVector('testdata.txt')
    testlabels = fileToVector('testlabels.txt')
    stoplist = fileToVector('stoplist.txt')
    return traindata, trainlabels, testdata, testlabels, stoplist


def readAndPreprocess(dryrun=False, classifier=None):
    traindata, trainlabels, testdata, testlabels, stoplist = read_data()
    trainlabels = np.array(trainlabels).astype(int)
    testlabels = np.array(testlabels).astype(int)
    
    return traindata, trainlabels, testdata, testlabels, stoplist


def build_vocabulary(traindata, stoplist):
    vocab = []
    for row in traindata:
        words = row.split()
        for word in words:
            if word not in vocab and word not in stoplist:
                vocab.append(word)
                
    vocab.sort()
    
    #print(vocab)
    return vocab


def build_vocab_map(vocab):
    vocab_map = {}
    for i in range(len(vocab)):
        vocab_map[vocab[i]] = i
        
    return vocab_map


def create_feature_vectors(data, vocab_map, M, stoplist):
    X = np.zeros((len(data),M))
    for i in range(len(data)):
        words = data[i].split()
        for word in words:
            if word not in stoplist:
                j = vocab_map.get(word) # j is the index of the word in the vocabulary
                if j is not None: #j is None for unseen data, we can safely ignore them as they create equal likelihood for the positive and negative classes 
                    X[i, j] = 1
    
    return X
    
