#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 14 11:02:39 2017

@author: anr.putina
"""

import sys
import pandas as pd

sys.path.append('../Functions')

import matplotlib
import matplotlib.pyplot as plt
matplotlib.style.use('ggplot')

from data_manipulation import *
from linear_regressor import LinearRegressor
from graphics import Graphics
g = Graphics()

feature = 'Jitter(%)'
file_name = 'parkinsons_updrs.data'

df = load_data(file_name)
df = remove_columns(df, 'age', 'sex', 'test_time', 'subject#' )

df = normalize_matrix(df)

k = 10

from scipy.spatial.distance import pdist, squareform

distances = pdist(df.values, metric='euclidean')
DistMatrix = squareform(distances)

k_nearest = []

for record in DistMatrix:
    
#    record.sort()
    distances = record[0:k]
    score = sum(distances)/k
    k_nearest.append(score)    
    
    
#k_nearest.sort(reverse=True)

plt.figure(figsize=(13,6))
plt.plot(k_nearest)

positions = []
for position, item in enumerate(k_nearest):
    if item > 8:
        print position
        positions.append(position)