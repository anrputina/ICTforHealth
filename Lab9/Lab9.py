#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 11 12:38:35 2017

@author: anr.putina
"""

import numpy as np
import scipy
import pandas as pd

from sklearn import svm
from sklearn import metrics
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.model_selection import ShuffleSplit

from classifier import Classifier


import matplotlib.pyplot as plt

def standardize (x):
    return (x-x.mean())/x.std()

def square_distance (x, xk):
    return np.linalg.norm(x - xk)**2

def cov_matrix (x):
    n = float(len(x))
    return 1.0/n * x.T.dot(x)

def load_data (full_classes=False):

    df = pd.read_csv("../Data/arrhythmia.data", header=None)
    df = df.replace({"?": np.NaN}).dropna(axis=1, how="any")

    if not full_classes:
        df.ix[df.iloc[:, -1] > 1, df.columns[-1]] = 2
             
    df = df.loc[:,(df!=0).any()]
    
    df_notnorm = df.copy()
    df.iloc[:, :-1] = df.iloc[:, :-1].apply(standardize)
    return df_notnorm, df

df_notnorm, df = load_data(False)

def get_PCA (x):
                    
    Rx = cov_matrix(x)
    eigvals, U = np.linalg.eig(Rx)
    L = len(np.where(eigvals.cumsum() < eigvals.sum() * 0.9)[0])    
    U = U[:, :L]            
    z = x.dot(U)
    z = z/z.std()
    
    return pd.concat([z, df.iloc[:, -1]], axis=1)

x = get_PCA(df.iloc[:, :-1])
y = df.iloc[:, -1]            

### SUPPORT VECTOR 
supportvector = svm.LinearSVC(C=2)

### SPLITTING DATASET
crossvalidation = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)

recalls = []
precisions = []
specificities = []

for train_index, test_index in crossvalidation.split(df):
    
    supportvector.fit(x.loc[train_index], y.loc[train_index])
    predicted = supportvector.predict(x.loc[test_index])
    
    classifier = Classifier()
    recall, precision, specificity = classifier.get_detections(predicted, y.loc[test_index])
    recalls.append(recall)
    precisions.append(precision)
    specificities.append(specificity)

print 'Recall: {}'.format(sum(recalls)/float(len(recalls)))
print 'Precision: {}'.format(sum(precisions)/float(len(precisions)))
print 'Specificity: {}'.format(sum(specificities)/float(len(specificities)))