#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 11 11:30:07 2017

@author: anr.putina
"""

import sys
import pandas as pd
import numpy as np

from sklearn.metrics import r2_score
from sklearn.metrics import median_absolute_error
from sklearn.metrics import mean_absolute_error, mean_squared_error

import tensorflow as tf

import matplotlib
import matplotlib.pyplot as plt
matplotlib.style.use('ggplot')

from sklearn.metrics import mean_absolute_error, mean_squared_error

sys.path.append('./../Functions')
from data_manipulation import *

file_name = 'parkinsons_updrs.data'

df = load_data(file_name)
df = df.groupby(["subject#", "test_time"]).mean()
df['subject#'] = list(df.index.get_level_values('subject#'))
df = remove_columns(df, 'age', 'sex')

training_df = select_training_df(df, 7, 42)
testing_df = select_testing_df(df, 1, 6)

training_df = normalize_matrix(training_df).dropna().drop('subject#', axis=1)
testing_df = normalize_matrix(testing_df).dropna().drop('subject#', axis=1)


#feature = "Jitter(%)"
feature = "motor_UPDRS"

y_train = training_df[feature]
training_df = training_df.drop(feature, axis=1)
y_test = testing_df[feature]
testing_df = testing_df.drop(feature, axis=1)

tf.set_random_seed(10000007)

N = training_df.shape[0] 
F = training_df.shape[1]


nh1 = 10
nh2 = 5
learning_rate = 1e-4

x = tf.placeholder(tf.float32, [None, F])
t = tf.placeholder(tf.float32, [None, 1])

#--- neural netw structure:
w1 = tf.Variable(tf.random_normal(shape=[F, nh1], mean=0.0, stddev=1.0, dtype=tf.float32, name="weights1"))
b1 = tf.Variable(tf.random_normal(shape=[1, nh1], mean=0.0, stddev=1.0, dtype=tf.float32, name="biases1"))
a1 = tf.matmul(x, w1) + b1
z1 = tf.nn.tanh(a1)

w2 = tf.Variable(tf.random_normal([nh1, nh2], mean=0.0, stddev=1.0, dtype=tf.float32, name="weights2"))
b2 = tf.Variable(tf.random_normal([1, nh2], mean=0.0, stddev=1.0, dtype=tf.float32, name="biases2"))
a2 = tf.matmul(z1, w2) + b2
z2 = tf.nn.tanh(a2)

w3 = tf.Variable(tf.random_normal([nh2, 1], mean=0.0, stddev=1.0, dtype=tf.float32, name="weights3"))
b3 = tf.Variable(tf.random_normal([1, 1], mean=0.0, stddev=1.0, dtype=tf.float32, name="biases3"))
y = tf.matmul(z2, w3) + b3   

#--- optimizer structure
cost = tf.reduce_sum(tf.squared_difference(y, t, name="objective_function"))
optim = tf.train.AdamOptimizer(learning_rate, name="GradientDescent")
optim_op = optim.minimize(cost, var_list=[w1, b1, w2, b2, w2, b3])
cost_history = []

#--- initialize
init = tf.global_variables_initializer()

#--- run the learning machine
sess = tf.Session()
sess.run(init)

for i in range(10000):
    # generate the data
    xval = training_df
    tval = y_train.values.reshape(N, 1)
    # train
    train_data = {x: xval, t: tval}
    sess.run(optim_op, feed_dict = train_data)
    cost_history += [cost.eval(feed_dict = train_data, session=sess)]   


#meanMI = []
#iterations = 0         
#costo = 10000
#for i in range(100000):
#    xval = training_df
#    tval = y_train.values.reshape(N, 1)
#    # train
#    train_data = {x: xval, t: tval}
#    sess.run(optim_op, feed_dict = train_data)
#    costo = cost.eval(feed_dict = train_data, session=sess)
#    if iterations % 1000 == 0:
#        print 'cost '+str(costo)
#        feed_dict = {x: testing_df}
#        yhat_test2 = sess.run(y, feed_dict)
#        meanMI.append(mean_squared_error(y_test, yhat_test2))
#        print mean_squared_error(y_test, yhat_test2)
#    cost_history += [costo]  
#    iterations += 1
    

#--- print the final results
w1 = sess.run(w1)
w2 = sess.run(w2)
w3 = sess.run(w3)
yhat_train2 = y.eval(feed_dict = train_data, session=sess)                
feed_dict = {x: testing_df}
yhat_test2 = sess.run(y, feed_dict)

fig, ax = plt.subplots()
ax.plot(y_test.values)
ax.plot(yhat_test2)

plt.figure()
plt.plot(cost_history)



import numpy as np

k_folds = 7
mean_error = []
    
for testing_number in range(k_folds):

    testing_group = set(range(testing_number*6 + 1, testing_number*6+6 + 1))     
    train_group = set(range(1,43)).difference(testing_group)
    
    training_df = select_training_df_bySubject(df, train_group)
    testing_df = select_training_df_bySubject(df, testing_group)

    y_train = training_df[feature]
    training_df = training_df.drop(feature, axis=1).drop('subject#', axis=1)
    y_test = testing_df[feature]
    testing_df = testing_df.drop(feature, axis=1).drop('subject#', axis=1)
    
    training_df = normalize_matrix(training_df).dropna()
    testing_df = normalize_matrix(testing_df).dropna()
    
    ### NEURAL NETWORK

    tf.set_random_seed(10000007)

    N = training_df.shape[0] 
    F = training_df.shape[1]
    
    
    nh1 = 18
    nh2 = 10
    learning_rate = 1e-4
    
    x = tf.placeholder(tf.float32, [None, F])
    t = tf.placeholder(tf.float32, [None, 1])
    
    #--- neural netw structure:
    w1 = tf.Variable(tf.random_normal(shape=[F, nh1], mean=0.0, stddev=1.0, dtype=tf.float32, name="weights1"))
    b1 = tf.Variable(tf.random_normal(shape=[1, nh1], mean=0.0, stddev=1.0, dtype=tf.float32, name="biases1"))
    a1 = tf.matmul(x, w1) + b1
    z1 = tf.nn.tanh(a1)
    
    w2 = tf.Variable(tf.random_normal([nh1, nh2], mean=0.0, stddev=1.0, dtype=tf.float32, name="weights2"))
    b2 = tf.Variable(tf.random_normal([1, nh2], mean=0.0, stddev=1.0, dtype=tf.float32, name="biases2"))
    a2 = tf.matmul(z1, w2) + b2
    z2 = tf.nn.tanh(a2)
    
    w3 = tf.Variable(tf.random_normal([nh2, 1], mean=0.0, stddev=1.0, dtype=tf.float32, name="weights3"))
    b3 = tf.Variable(tf.random_normal([1, 1], mean=0.0, stddev=1.0, dtype=tf.float32, name="biases3"))
    y = tf.matmul(z2, w3) + b3   
    
    #--- optimizer structure
    cost = tf.reduce_sum(tf.squared_difference(y, t, name="objective_function"))
    optim = tf.train.AdamOptimizer(learning_rate, name="GradientDescent")
    optim_op = optim.minimize(cost, var_list=[w1, b1, w2, b2, w2, b3])
    cost_history = []
        
    init = tf.global_variables_initializer()
    
    sess = tf.Session()
    sess.run(init)
    xval = training_df.values
    tval = y_train.values.reshape(N, 1)
    
    for i in range(10000):
        train_data = {x: xval, t: tval}
        sess.run(optim_op, feed_dict = train_data)
        c = cost.eval(feed_dict = train_data, session=sess)
        cost_history += [c]
    
    #--- print the final results
    print(sess.run(w1), sess.run(b1))
    a = sess.run(w1)
    yhat_train = y.eval(feed_dict = train_data, session=sess)
    feed_dict = {x: testing_df}
    yhat_test = sess.run(y, feed_dict)
    
    error = mean_squared_error(y_test.tolist(), yhat_test)
    mean_error.append(error)
    print error    
    tf.reset_default_graph()

print 'Mean Error: ' + str(sum(mean_error)/len(mean_error))