#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 17 14:58:17 2017

@author: anr.putina
"""

import sys
import pandas as pd
import numpy as np

#sys.path.append('/Users/anr.putina/Desktop/TELemetry/Architecture')
sys.path.append('../Functions')
from data_manipulation import *
from classifier import Classifier


from classifier import Classifier
def normalize_matrix(df):
    return (df - df.mean())/df.std()

file_name = 'arrhythmia.data'
df = load_arrhythmia_data(file_name, clusters='boolean')

class_id = df['class']
classes = range(1,df['class'].max()+1)

df = df.drop('class', axis=1)

############### CLUSTERING
from sklearn.neighbors import NearestNeighbors


dataset = normalize_matrix(df).dropna(axis=1)

from sklearn.cluster import DBSCAN
db = DBSCAN(eps=9.5, min_samples=4, algorithm='brute').fit(dataset)
a = db.labels_
a[a==0] = 1
a[a==-1] = 2


classifier = Classifier()
classifier.check_detections(a, class_id)