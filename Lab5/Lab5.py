#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  3 16:31:21 2017

@author: anr.putina
"""

import sys
import pandas as pd
import numpy as np

import matplotlib
import matplotlib.pyplot as plt
matplotlib.style.use('ggplot')

sys.path.append('../Functions')
from data_manipulation import *
from classifier import Classifier

def standardize (x):
    return (x-x.mean())/x.std()

def square_distance (x, xk):
    return np.linalg.norm(x - xk)**2

def cov_matrix (x):
    n = float(len(x))
    return 1.0/n * x.T.dot(x)

from classifier import Classifier

df = pd.read_csv('./../Data/chronic_kidney_disease.csv', skipinitialspace=True, header=None)

df = df.replace({"?": np.NaN})
df = df.replace({"\t?": np.NaN})
df = df.replace({"\tyes": "yes"})
df = df.replace({"\tno": "no"})
df = df.replace({"ckd\t": "no"})

for col in df:
    df[col] = df[col].fillna(df[col].mode().values[0])
    
keylist = ["normal","abnormal","present","notpresent","yes","no","good","poor","ckd","notckd"]
keymap = [0,1,0,1,0,1,0,1,1,0]
df = df.replace(keylist, keymap)

for col in [5,6,7,8,18,21,22,23]:
    df[(-df[col].isin([0,1]))] = df[col].mode().values[0]
    
true = df[24]
df = df.drop(24, axis=1)

from scipy.cluster.hierarchy import dendrogram, linkage, fcluster

df = df.apply(pd.to_numeric)

Z = linkage(df.apply(standardize).values, 
            method='single', 
            metric='euclidean')

### PLOT DENDOGRAM
fig, ax = plt.subplots(figsize=(13, 6))
d1 = dendrogram(Z)
ax.patch.set_facecolor('white')
ax.grid(c='grey')
ax.get_xaxis().set_visible(False)

#plt.figure(figsize=(13, 6))
#plt.title("Reduced dendrogram, Mathematica(TM) style")
#d2 = dendrogram(Z, truncate_mode="mtica")
#
#plt.figure(figsize=(13, 6))
#plt.title("Reduced dendrogram, Last p method")
#d3 = dendrogram(Z, truncate_mode="lastp")

from sklearn import cluster
from sklearn import tree
from sklearn import metrics
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.model_selection import ShuffleSplit

model = cluster.AgglomerativeClustering(n_clusters=2, linkage='complete')
clusters = model.fit_predict(df)

true_positive = ( (clusters==1) & (true == 1)).sum() / float((true == 1).sum())
true_negative = ( (clusters==0) & (true == 0)).sum() / float((true == 0).sum())
false_positive = ( (clusters==1) & (true == 0)).sum() / float((true == 0).sum())
false_negative = ( (clusters==0) & (true == 1)).sum() / float((true == 1).sum())

print 'Precision: {}'.format(true_positive/(true_positive+false_positive))
print 'Recall: {}'.format(true_positive/(true_positive+false_negative))
print 'Specificity: {}'.format(true_negative/(true_negative+false_positive))

from sklearn.tree import DecisionTreeClassifier
decisionTree = DecisionTreeClassifier()

crossvalidation = ShuffleSplit(n_splits=10, test_size=0.20, random_state=0)
crossvalidation.get_n_splits(df)

recalls = []
precisions = []
specificities = []

for train_index, test_index in crossvalidation.split(df):
    
    decisionTree.fit(df.loc[train_index], true.loc[train_index])
    predicted = decisionTree.predict(df.loc[test_index])
    
    classifier = Classifier()
    recall, precision, specificity = classifier.get_detections_NN(true.loc[test_index], predicted)
    recalls.append(recall)
    precisions.append(precision)
    specificities.append(specificity)
    
print 'Recall: {}'.format(sum(recalls)/float(len(recalls)))
print 'Precision: {}'.format(sum(precisions)/float(len(precisions)))
print 'Specificity: {}'.format(sum(specificities)/float(len(specificities)))

tree.export_graphviz(decisionTree, out_file='tree.dot')