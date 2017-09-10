#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 19 17:07:16 2017

@author: anr.putina
"""

import sys
import pandas as pd
import numpy as np

import matplotlib
import matplotlib.pyplot as plt
matplotlib.style.use('ggplot')

#sys.path.append('/Users/anr.putina/Desktop/TELemetry/Architecture')
sys.path.append('../Functions')
from data_manipulation import *

from classifier import Classifier


file_name = 'arrhythmia.data'
df = load_arrhythmia_data(file_name, clusters='boolean')

class_id = df['class']
classes = range(1,df['class'].max()+1)

df = df.drop('class', axis=1)

k = 2
classifier = Classifier()
classification = classifier.kmeans(df, k, class_id=class_id, start='random')
classifier.check_detections(classification, class_id)

classification = classifier.kmeans(df, k, class_id=class_id, start='mean')
classifier.check_detections(classification, class_id)