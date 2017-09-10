#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 14 17:33:28 2017

@author: anr.putina
"""

import sys
import pandas as pd
import numpy as np

sys.path.append('../Functions')
from data_manipulation import *

from classifier import Classifier

from linear_regressor import LinearRegressor
from graphics import Graphics
g = Graphics()

import matplotlib
import matplotlib.pyplot as plt
matplotlib.style.use('ggplot')

from data_manipulation import *
from scipy.spatial.distance import euclidean

file_name = 'arrhythmia.data'

df2 = load_arrhythmia_data(file_name, clusters='boolean')

mag = df2 != 0
zeros = mag.sum()

class_id = df2['class']
classes = range(1,df2['class'].max()+1)

df2 = df2.drop('class', axis=1)
df2 = normalize_matrix(df2)

classifier = Classifier()

w1 = df2[class_id==1].mean()
w2 = df2[class_id==2].mean()

def square_distance(x, xk):
    return np.linalg.norm(x-xk)**2

df_min_dist = classifier.min_dist_classification(df2, class_id, classes)
classifier.check_detections(df_min_dist['previsione'], class_id)

df_min_dist_pca = classifier.pca_classification(df2, class_id, classes, classification = 'min_dist')
classifier.check_detections(df_min_dist_pca['previsione'], class_id)

df_distance_bayes = classifier.pca_classification(df2, class_id, classes, classification = 'distance_bayes')
classifier.check_detections(df_distance_bayes['previsione'], class_id)

df_region_bayes = classifier.pca_classification(df2, class_id, classes, classification = 'region_bayes')
classifier.check_detections(df_region_bayes['previsione'], class_id)