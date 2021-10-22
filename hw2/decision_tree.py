# -*- coding: utf-8 -*-
"""
@author: Abdullah Mamun
"""
import numpy as np
from math import log2
from copy import copy,deepcopy
verbose = False
class Node:
    def __init__(self):
        self.label = None
        self.left = None
        self.right = None
        self.splitfeature = None
        self.threshold = None
        self.parent = None
        self.isleaf = None
        self.plus = None
        self.minus = None
    


def entropy(Y):
    n = len(Y)
    plus = np.count_nonzero(Y)
    minus = n - plus
    if plus == 0:
        plusE = 0
    else:
        plusE = -plus/n*log2(plus/n)
    if minus == 0:
        minusE = 0
    else:
        minusE = - minus/n*log2(minus/n)
    return  plusE + minusE

def find_ig(X, Y, feature, threshold):
    #print(X.shape)
    #print(X[:, feature].shape)
    #print(X[:, feature])
    global verbose
    X_col = X[:,feature]
    if verbose:
        print(X_col.shape)
        print(X_col)
    
    indices = np.where(X_col<threshold)[0]
    indices2 = np.where(X_col>=threshold)[0]
    
    if verbose:
        print(indices)
        print(indices2)
    n = len(Y)
    
    Y_below = Y[indices]
    Y_above = Y[indices2] 
    
    E_feature= entropy(Y)
    E_below = entropy(Y_below)
    E_above = entropy(Y_above)
    if verbose:
        print("printing Entropies")
        print(n, len(Y_below), len(Y_above))
        print(E_feature, E_below, E_above)
    
    return E_feature - len(Y_below)/n*E_below - len(Y_above)/n*E_above # E_feature is the entropy of the feature, and the other terms together are the
                                                 #entropy of feature|feature value

def argmax_ig(X, Y, thresholdlist):
    max_ig = -1
    argmax_feature = None
    argmax_threshold = None
    for (feature, threshold) in thresholdlist:
        
        ig = find_ig(X, Y, feature, threshold)
        #print(feature, threshold, ig)
        if ig > max_ig:
            argmax_feature = feature
            argmax_threshold = threshold
            max_ig = ig
            
            
    #print(argmax_feature, argmax_threshold)
    return argmax_feature, argmax_threshold
    
def ID3(parent, X, Y, thresholdlist):
    root = Node()
    root.parent = parent
    n = len(Y)
    
    if n ==0: #no examples, so prior is our only hope
        root.label = 0 #prior belief, the most frequent label in the example: benign
        root.isleaf = True
        return root
        
    plus = np.count_nonzero(Y)
    minus = n-plus
    
    
    if n == plus:  #if all examples are positive
        root.label = 1
        root.isleaf = True
        return root
    elif n == minus:  #if all examples are negative
        root.label = 0
        root.isleaf = True
        return root

    if len(thresholdlist) == 0:  #no more attribute left, then return the label of majority
        if plus >=minus:
            root.label = 1
        else:
            root.label = 0
        root.isleaf = True
        return root
    
    root.plus = plus
    root.minus = minus
    
    feature, threshold = argmax_ig(X, Y, thresholdlist)
    root.splitfeature = feature
    root.threshold = threshold
    root.isleaf = False

    X_col = X[:,feature]
    indices = np.where(X_col<threshold)[0]
    indices2 = np.where(X_col>=threshold)[0]
    
    X_below = X[indices]
    X_above = X[indices2]    
    Y_below = Y[indices]
    Y_above = Y[indices2]
    
    tempthresholdlist = copy(thresholdlist)
    tempthresholdlist.remove((feature, threshold))
    global verbose
    verbose = False
    root.left = ID3(root, X_below, Y_below, tempthresholdlist)
    root.right = ID3(root, X_above, Y_above, tempthresholdlist)
    


    return root    
    
    
    
def create_decision_tree(X, Y):
    features = [i for i in range(9)] #features are 0 to 8
    thresholdlist = []
    thresholdranges  = [i+0.5 for i in range(1,10)] #2.3a, here we create the thresholdrange
    for feature in features:
        for threshold in thresholdranges:
            thresholdlist.append((feature, threshold))
            
    tree = ID3(None, X, Y, thresholdlist)
    return tree
    


def predict_single(tree, x):
    temp = tree
    #print(tree)
    #print(temp)
    while temp.isleaf is False:
        if x[temp.splitfeature] < temp.threshold:
            temp = temp.left
        else:
            temp = temp.right
        #print(temp)
    
    return temp.label

def predict(tree, X):
    n= X.shape[0]
    Ypred = [0 for i in range(n)]
    for i in range(n):
        Ypred[i] = predict_single(tree, X[i])
    
    return Ypred


def evaluate(tree, X, Y):
    Ypred = predict(tree, X)
    acc = np.count_nonzero(Y==Ypred)/len(Y)
    return acc    
    
def run(Xtrain, Ytrain, Xval, Yval, Xtest, Ytest, problemNo):
    print("Running experiment for", problemNo)
    
    print("Building the tree..")
    tree = create_decision_tree(Xtrain, Ytrain) #tree construction is done
    print("Tree has been built")
    valacc = evaluate(tree, Xval, Yval)
    testacc = evaluate(tree, Xtest, Ytest)
    
    print("Val acc:",valacc, 'Test acc:',testacc)
    
    
    
    

def copy_tree(tree):
    if tree is None:
        return None
    
    tree2 = deepcopy(tree)
    tree2.left = copy_tree(tree.left)
    tree2.right = copy_tree(tree.right)
    return tree2
    
def prune_tree(tree, subtree, Xval, Yval, bestvalacc, path):
    
    if subtree.isleaf:
        return tree, bestvalacc
    tree, bestvalacc = prune_tree(tree, subtree.left, Xval, Yval, bestvalacc, path+"->L")
    tree, bestvalacc = prune_tree(tree, subtree.right, Xval, Yval, bestvalacc, path+"->R")
    
    #tree2 = copy_tree(tree)
    subtree.isleaf = True
    subtree.label = int(subtree.plus >=subtree.minus)
    
    valacc = evaluate(tree, Xval, Yval)
    if valacc > bestvalacc:  #pruning producing good result
        print("==IMPROVEMENT==\nFound better tree by pruning the branch below ", path)
        print("Best val acc updated: from {} to {}".format(bestvalacc, valacc))
        bestvalacc = valacc
    
    else: #did not find a better result, so rollback the change
        subtree.isleaf = False
        subtree.label = None
        
    
    return tree, bestvalacc
    
        
    
    

def run_with_pruning(Xtrain, Ytrain, Xval, Yval, Xtest, Ytest, problemNo):
    print()
    print()
    print("Running experiment for", problemNo)
    
    print("Building the tree..")
    tree = create_decision_tree(Xtrain, Ytrain) #tree construction is done
    print("Tree has been built")
    valacc = evaluate(tree, Xval, Yval)
    print("Starting the pruning process by recursion...")
    print()
    
    final_tree, _ = prune_tree(tree, tree, Xval, Yval, valacc, "root")
    print("Pruning is complete")
    valacc = evaluate(final_tree, Xval, Yval)
    testacc = evaluate(final_tree, Xtest, Ytest)
    
    print()
    print("Final metrics after pruning")
    print("Val acc:",valacc, 'Test acc:',testacc)
    
    