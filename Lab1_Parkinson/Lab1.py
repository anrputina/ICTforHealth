#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 10 17:00:39 2017

@author: anr.putina
"""

import sys
import pandas as pd

from sklearn.metrics import r2_score
from sklearn.metrics import median_absolute_error
from sklearn.metrics import mean_absolute_error, mean_squared_error

sys.path.append('../Functions')
from data_manipulation import *
from linear_regressor import LinearRegressor
from graphics import Graphics
g = Graphics()

from math import sqrt

import matplotlib
import matplotlib.pyplot as plt
matplotlib.style.use('ggplot')

#feature = 7 - 1
feature = 'Jitter(%)'
#feature = 'motor_UPDRS'
file_name = 'parkinsons_updrs.data'

df = load_data(file_name)
df = df.groupby(["subject#", "test_time"]).mean()
df['subject#'] = list(df.index.get_level_values('subject#'))
df = remove_columns(df, 'age', 'sex')

training_df = select_training_df(df, 1, 36)
testing_df = select_testing_df(df, 37, 42)

training_df = training_df.drop('subject#', axis=1)
testing_df = testing_df.drop('subject#', axis=1)

training_df = normalize_matrix(training_df)
testing_df = normalize_matrix(testing_df)

y_train = select_y_byName(training_df, feature)
y_test = select_y_byName(testing_df, feature)

x_train = select_x_byName(training_df, feature)
x_test = select_x_byName(testing_df, feature)

### REGRESSOR OBJECT
regressor = LinearRegressor()

### TRAIN MSE - GRADIENT - STEEPEST
regressor.train_mse(x_train, y_train)
regressor.train_gradient(x_train.values, y_train.values, gamma=1.0e-5, epsilon=1.0e-4)
regressor.train_stepeest_descent(x_train.values, y_train.values, epsilon=1.0e-4)

### ESTIMATE VALUES TRAIN AND TEST
estimated_values_train_mse = regressor.estimate_values(x_train, regressor.mse_parameters)
estimated_values_train_gradient = regressor.estimate_values(x_train, regressor.gradient_parameters)
estimated_values_train_steepest = regressor.estimate_values(x_train, regressor.stepeest_descent_parameters)

estimated_values_test_mse = regressor.estimate_values(x_test, regressor.mse_parameters)
estimated_values_test_gradient = regressor.estimate_values(x_test, regressor.gradient_parameters)
estimated_values_test_steepest = regressor.estimate_values(x_test, regressor.stepeest_descent_parameters)

### CONFIGURATION PLOTS - MATPLOTLIB
labels = list(x_train.columns)
xticks = range(len(x_train.columns))
fontsz = 20
fontlabel = 20
matplotlib.rcParams['xtick.labelsize'] = fontlabel 
matplotlib.rcParams['ytick.labelsize'] = fontlabel 
matplotlib.rc('axes', edgecolor='black')

### PLOT WEIGHTS
fig, ax = plt.subplots(figsize=(13,6))
ax.plot(abs(regressor.mse_parameters), linewidth=2.0, label='MSE', marker='o')
ax.plot(abs(regressor.gradient_parameters), linewidth=2.0, label='gradient', marker='d')
ax.plot(abs(regressor.stepeest_descent_parameters), linewidth=2.0, label='stepeest descent', marker='x')
ax.set_ylabel('values', fontsize=fontsz)
ax.set_yscale('log')
ax.patch.set_facecolor('white')
ax.grid(c='gray')
legend = plt.legend(loc=2, fontsize=20)
legend.get_frame().set_alpha(0.5)
ax.set_xlim(0,15)
plt.xticks(xticks, labels, rotation='vertical')
plt.subplots_adjust(bottom=0.15)


#### ESTIMATED TRAIN PLOT
#fix, ax = plt.subplots(figsize=(13,6))
#ax.plot(y_train.tolist(), label='true', color='black', linestyle='--')
#ax.plot(estimated_values_train_mse, label='MSE', linestyle='--')
#ax.plot(estimated_values_train_gradient, label='gradient', linestyle='--')
#ax.plot(estimated_values_train_steepest, label='stepeest descent', linestyle='--')
#ax.patch.set_facecolor('white')
#ax.grid(c='gray')
#legend = plt.legend(loc=2, ncol=2, fontsize=20)
#legend.get_frame().set_alpha(0.5)
#ax.set_xlabel('#sample', fontsize=fontsz)
#ax.set_ylabel(feature + ' normalized', fontsize=fontsz)
#ax.set_xlim(0, 844)


#### ESTIMATED TEST PLOT
#fix, ax = plt.subplots(figsize=(13,6))
#ax.plot(y_test.tolist(), label='true', color='black')
#ax.plot(estimated_values_test_mse, label='MSE')
#ax.plot(estimated_values_test_gradient, label='gradient')
#ax.plot(estimated_values_test_steepest, label='stepeest descent')
#ax.patch.set_facecolor('white')
#ax.grid(c='gray')
#legend = plt.legend(loc=1, ncol=2, fontsize=20)
#legend.get_frame().set_alpha(0.5)
#ax.set_xlabel('#sample', fontsize=fontsz)
#ax.set_ylabel(feature + ' normalized', fontsize=fontsz)
#ax.set_xlim(0, 151)


#### SCATTER TRAIN 
#fig, ax = plt.subplots(figsize=(13,6))
#ax.scatter(y_train, estimated_values_train_mse, label='MSE', alpha=0.5)
#ax.scatter(y_train, estimated_values_train_gradient, label='gradient', color='r', alpha=0.5)
#ax.scatter(y_train, estimated_values_train_steepest, label='stepeest descent', color='green', alpha=0.5)
#ax.patch.set_facecolor('white')
#ax.grid(c='gray')
#legend = plt.legend(loc=2, ncol=2, fontsize=20)
#legend.get_frame().set_alpha(0.5)
#ax.plot(range(-3, 21), range(-3, 21), color='black')
#ax.set_xlim(-5, 20)
#ax.set_xlabel('y true')
#ax.set_ylabel('y hat')


####SCATTER TEST
#fig, ax = plt.subplots(figsize=(13,6))
#ax.scatter(y_test.values, estimated_values_test_mse, label='MSE', alpha=0.5)
#ax.scatter(y_test.values, estimated_values_test_gradient, label='gradient', color='r', alpha=0.5)
#ax.scatter(y_test.values, estimated_values_test_steepest, label='stepeest descent', color='green', alpha=0.5)
#ax.patch.set_facecolor('white')
#ax.grid(c='gray')
#legend = plt.legend(loc=2, ncol=2, fontsize=20)
#legend.get_frame().set_alpha(0.5)
#ax.plot(range(-2, 8), range(-2, 8), color='black')
#ax.set_xlim(-2, 8)
#ax.set_ylim(-2, 8)
#ax.set_xlabel('y true')
#ax.set_ylabel('y hat')


#### HIST ERROR TRAIN
#bins=50
#fig, ax = plt.subplots(figsize=(13,6))
#ax.hist(y_train - estimated_values_train_mse, label='MSE', alpha=0.5, bins=bins, color='blue')
#ax.hist(y_train - estimated_values_train_gradient, label='gradient', color='r', alpha=0.5, bins=bins)
#ax.hist(y_train - estimated_values_train_steepest, label='stepeest descent', color='green', alpha=0.5, bins=bins)
##ax.hist(y_train - yhat_train.flatten(), label='neural network', color='grey', alpha=0.5, normed=True, bins=bins)
##ax.hist(y_train - yhat_train2.flatten(), label='neural network hd', color='grey', alpha=0.5, normed=True, bins=bins)
#ax.patch.set_facecolor('white')
#ax.grid(c='gray')
#legend = plt.legend(loc=1, ncol=1, fontsize=20)
#legend.get_frame().set_alpha(0.5)
#ax.set_ylabel('count', fontsize=20)
#ax.set_xlabel('error', fontsize=20)


#### HIST ERROR TEST
#bins=50
#fig, ax = plt.subplots(figsize=(13,6))
#ax.hist(y_test - estimated_values_test_mse, label='MSE', alpha=0.5, bins=bins, color='blue')
#ax.hist(y_test - estimated_values_test_gradient, label='gradient', color='r', alpha=0.5, bins=bins)
#ax.hist(y_test - estimated_values_test_steepest, label='stepeest descent', color='green', alpha=0.5, bins=bins)
#ax.patch.set_facecolor('white')
#ax.grid(c='gray')
#legend = plt.legend(loc=1, ncol=1, fontsize=20)
#legend.get_frame().set_alpha(0.5)
#ax.set_ylabel('count', fontsize=20)
#ax.set_xlabel('error', fontsize=20)

### CROSS VAL - KFOLD

import numpy as np

k_folds = 7
r2_list=[]
mean_error = []

for testing_number in range(k_folds):

    testing_group = set(range(testing_number*6 + 1, testing_number*6+6 + 1))     
    train_group = set(range(1,43)).difference(testing_group)
    
    training_df = select_training_df_bySubject(df, train_group)
    testing_df = select_training_df_bySubject(df, testing_group)

    training_df = normalize_matrix(training_df)
    testing_df = normalize_matrix(testing_df)

    y_train = select_y_byName(training_df, feature)
    y_test = select_y_byName(testing_df, feature)

    x_train = select_x_byName(training_df, feature)
    x_test = select_x_byName(testing_df, feature)
    
    regressor.train_mse(x_train, y_train)
    estimated_values = regressor.estimate_values(x_test, regressor.mse_parameters)
    
#    regressor.train_gradient(x_train.values, y_train.values, gamma=1.0e-5, epsilon=1.0e-4)
#    estimated_values = regressor.estimate_values(x_test, regressor.gradient_parameters)    

#    regressor.train_stepeest_descent(x_train.values, y_train.values, epsilon=1.0e-4)
#    estimated_values = regressor.estimate_values(x_test, regressor.stepeest_descent_parameters)


    r2 = r2_score(y_test.tolist(), estimated_values) 
    error = mean_squared_error(y_test.tolist(), estimated_values)
    r2_list.append(r2)
    mean_error.append(error)
    print r2, error    


print 'Mean R2 Score: ' + str(sum(r2_list)/len(r2_list))
print 'Mean Error: ' + str(sum(mean_error)/len(mean_error))

### FEATURE INSPECTION - FEATURE COMPARISON

#norm = normalize_matrix(df)
##fontsz = 20
##fontlabel = 16
##matplotlib.rcParams['xtick.labelsize'] = fontlabel 
##matplotlib.rcParams['ytick.labelsize'] = fontlabel 
##matplotlib.rc('axes', edgecolor='black')
##
#COMPARISON = 'Jitter(%)'
#
#fig, ax = plt.subplots(1, 2, figsize=(13,6))
#ii=0
#for feature in ['Jitter:RAP', 'motor_UPDRS']:
#    ax[ii].scatter(norm[COMPARISON].values, norm[feature].values, s=10)
#    ax[ii].set_ylabel(feature, fontsize=fontsz)
#    ax[ii].set_xlabel(COMPARISON, fontsize=fontsz)
#    ax[ii].patch.set_facecolor('white')
#    ax[ii].grid(c='gray')
#    ii += 1