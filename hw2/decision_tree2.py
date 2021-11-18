
import numpy
import math


class Cache:
    def __init__(self):
        pass
    
class Treenode:
    def __init__(self):
        pass

def mylog2(value):
    if value<=0:
        return 0
    else:
        return math.log2(value)
def DecisionTree(X, Y, allthresholds):
    curNode =Treenode()
    
    
    if X.shape[0] == 0:
        curNode.type = "leaf"
        curNode.label = 0
        return curNode
    
    if X.shape[0] == numpy.count_nonzero(Y):  
        curNode.type = "leaf"
        curNode.label = 1
        return curNode
    
    if 0 == numpy.count_nonzero(Y):  
        curNode.label = 0
        curNode.type = "leaf"
        return curNode
    
    if len(allthresholds) ==0:
        curNode.label = 0
        curNode.type = "leaf"
        return curNode
        

    best_ig = -100
    curNode.f = None
    curNode.thr = None
    for k in range(len(allthresholds)):
        f = allthresholds[k][0]
        thr = allthresholds[k][1]
        y1 = []
        y2 = []
        for i in range(X.shape[0]):
            if X[i, f] < thr:
                y1.append(Y[i])
            else:
                y2.append(Y[i])
        
        positive = numpy.count_nonzero(Y)
        negative = len(Y)-positive
        
        E1=  -positive/(len(Y))*mylog2(positive/(len(Y))) - negative/(len(Y))*mylog2(negative/(len(Y)))
        
        positive = numpy.count_nonzero(y1)
        negative = len(y1)-positive
        if len(y1) == 0:
            E2 = 0
        else:
            E2=  -positive/(len(y1))*mylog2(positive/(len(y1))) - negative/(len(y1))*mylog2(negative/(len(y1)))
        
        positive = numpy.count_nonzero(y2)
        negative = len(y2)-positive
        if len(y2) ==0:
            E3=0
        else:
            E3 = -positive/(len(y2))*mylog2(positive/(len(y2))) - negative/(len(y2))*mylog2(negative/(len(y2)))
        
        E_conditional = len(y1)/len(Y)*E2+ len(y2)/len(Y)*E3
        IG =  E1 - E_conditional
        print(IG)
        if IG > best_ig:
            curNode.f = f
            curNode.thr = thr
            best_ig = IG
            best_f = f
            best_thr = thr
    
            
    X1 = []
    X2 = []
    for i in range(X.shape[0]):
        if X[i][curNode.f] < curNode.f:
            X1.append(X[i])
            y1.append(Y[i])
        else:
            X2.append(X[i])
            y2.append(Y[i])
    
    X1 = numpy.asarray(X1)
    X2 = numpy.asarray(X2)
    thr_updated = []
    for i in range(len(allthresholds)):
        f1 = allthresholds[i][0]
        thr1 = allthresholds[i][1]
        
        if f1 == curNode.f and thr1 == curNode.thr:
            pass
        else:
            thr_updated.append((f1, thr1))
 
    if len(X1) ==0:
        curNode.left = Treenode()
        curNode.left.type = "leaf"
        curNode.left.label = 0
    else:
        curNode.left = DecisionTree(X1, y1, thr_updated)
        
    if len(X2) ==0:
        curNode.right = Treenode()
        curNode.right.type = "leaf"
        curNode.right.label = 0
    else:
        curNode.right = DecisionTree(X2, y2, thr_updated)
    
    curNode.type = "impure"
    positive = numpy.count_nonzero(Y)
    negative = len(Y)-positive
    if positive>=negative:
        curNode.pruningLabel = 1
    else:
        curNode.pruningLabel = 0
        
    return curNode    

def getAccuracy(tree, X, Y):
    curNode = tree
    count = 0
    for i in range(X.shape[0]):
        while curNode.type == "impure":
            if X[i][curNode.f] < curNode.thr:
                curNode = curNode.left
            else:
                curNode = curNode.right
        
        yhat = curNode.label
        if Y[i] == yhat:
            count+=1
    
    return count/X.shape[0]    
     

def recursive_pruning(T, cache):
    import copy
    if T.type == "leaf":
        return T
    
    T.left = recursive_pruning(T.left, cache)
    T.right = recursive_pruning(T.right, cache)
    
    T_prime = copy.deepcopy(T)
    T_prime.type = "leaf"
    T_prime.label = T_prime.pruningLabel
    
    if getAccuracy(T, cache.valX, cache.valY)< getAccuracy(T_prime, cache.valX, cache.valY):
        T = T_prime
        
    return T
  
def createAndTest(trainX, valX, testX, trainY, valY, testY):
    
    thresholds = [1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5]
    allthresholds = []
    for i in range(9):
        for j in range(len(thresholds)):
            allthresholds.append((i, thresholds[j]))
    tree = DecisionTree(trainX, trainY, allthresholds)

    
    print(getAccuracy(tree, valX, valY))
    print(getAccuracy(tree, testX, testY))
    
    cache = Cache()
    cache.valX = valX
    cache.valY = valY
    
    
    tree = recursive_pruning(tree, cache)
    print(getAccuracy(tree, valX, valY))
    print(getAccuracy(tree, testX, testY))
    
    
    

    
        