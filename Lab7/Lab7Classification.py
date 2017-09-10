#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 18 17:05:43 2017

@author: anr.putina
"""

import sys
import pandas as pd
import numpy as np

import tensorflow as tf

import matplotlib
import matplotlib.pyplot as plt
matplotlib.style.use('ggplot')

sys.path.append('../Functions')
from data_manipulation import *

from classifier import Classifier

def standardize (x):
    return (x-x.mean())/x.std()

from sklearn.metrics import mean_absolute_error, mean_squared_error


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


xks = df.groupby(df.columns[-1]).mean()
yy = df.iloc[:, :-1]
class_id = df.iloc[:, -1].replace({1:0}).replace({2:1})

N = len(yy)
F = len(yy.columns)


#--- initial settings
nh1 = 257
nh2 = 128

learning_rate = 3e-2
x = tf.placeholder(tf.float32, [N, F])#inputs
t = tf.placeholder(tf.float32, [N, 1])#desired outputs

#--- neural netw structure:
w1 = tf.Variable(tf.random_normal(shape=[F, nh1], mean=0.0, stddev=1.0, dtype=tf.float32, name="weights1"))
b1 = tf.Variable(tf.random_normal(shape=[1, nh1], mean=0.0, stddev=1.0, dtype=tf.float32, name="biases1"))
a1 = tf.matmul(x, w1) + b1
z1 = tf.nn.sigmoid(a1)

w2 = tf.Variable(tf.random_normal([nh1, nh2], mean=0.0, stddev=1.0, dtype=tf.float32, name="weights2"))
b2 = tf.Variable(tf.random_normal([1, nh2], mean=0.0, stddev=1.0, dtype=tf.float32, name="biases2"))
a2 = tf.matmul(z1, w2) + b2
z2 = tf.nn.sigmoid(a2)

w3 = tf.Variable(tf.random_normal([nh2, 1], mean=0.0, stddev=1.0, dtype=tf.float32, name="weights3"))
b3 = tf.Variable(tf.random_normal([1, 1], mean=0.0, stddev=1.0, dtype=tf.float32, name="biases3"))
y = tf.nn.sigmoid(tf.matmul(z2, w3) + b3)


#--- optimizer structure
cost = tf.reduce_sum(tf.squared_difference(y, t, name="objective_function"))
optim = tf.train.GradientDescentOptimizer(learning_rate, name="GradientDescent")
optim_op = optim.minimize(cost, var_list=[w1, b1, w2, b2, w2, b3])

#--- initialize
init = tf.global_variables_initializer()

#--- run the learning machine
sess = tf.Session()
sess.run(init)
for i in range(1000):
    # generate the data
    xval = yy
    tval = class_id.values.reshape(N, 1)
    # train
    train_data = {x: xval, t: tval}
    sess.run(optim_op, feed_dict = train_data)
    c = cost.eval(feed_dict = train_data, session=sess)
    
    
    
yhat_train = np.round(y.eval(feed_dict = train_data, session=sess))
yhat_train = np.array(yhat_train, dtype=np.int32).reshape(len(yhat_train),)

classifier = Classifier()
classifier.check_detections_NN(yhat_train, class_id)