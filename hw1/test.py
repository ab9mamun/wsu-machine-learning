# -*- coding: utf-8 -*-
"""
Created on Mon Sep 20 18:15:31 2021

@author: abdullahal.mamun1
"""
import pandas as pd

from datamanager import read_data


traindf, testdf = read_data()

print(traindf.head())
print(testdf.head())

train, test = traindf.values, testdf.values
Ytrain = train[:,0]
Xtrain = train[:,1:]

print(Xtrain.shape)
print(Ytrain[:5])


