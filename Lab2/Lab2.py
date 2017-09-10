#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 13 16:11:44 2017

@author: anr.putina
"""

import sys
import numpy as np
import pandas as pd
sys.path.append('../Functions')
from data_manipulation import *

from sklearn.metrics import r2_score
from sklearn.metrics import median_absolute_error
from sklearn.metrics import mean_absolute_error, mean_squared_error

from linear_regressor import LinearRegressor
from graphics import Graphics
g=Graphics()

import matplotlib
import matplotlib.pyplot as plt
matplotlib.style.use('ggplot')


#feature = 'motor_UPDRS'
feature = 'Jitter(%)'

file_name = 'parkinsons_updrs.data'

df = load_data(file_name)
df = df.groupby(["subject#", "test_time"]).mean()
df['subject#'] = list(df.index.get_level_values('subject#'))
df = remove_columns(df, 'age', 'sex')

training_df = select_training_df(df, 1, 36)
testing_df = select_testing_df(df, 37, 42)

training_df = normalize_matrix(training_df)
testing_df = normalize_matrix(testing_df)

y_train = select_y_byName(training_df, feature)
y_test = select_y_byName(testing_df, feature)

x_train = select_x_byName(training_df, feature)
x_test = select_x_byName(testing_df, feature)

regressor = LinearRegressor()

regressor.train_pca(x_train.values, y_train.values, L=10)

estimated_values_test = regressor.estimate_values(x_test, regressor.pca_parameters)
print r2_score(y_test.tolist(), estimated_values_test)
print mean_squared_error(y_test, estimated_values_test) 

#g.hist_errors(y_test.tolist(), estimated_values_test)
#g.test_pred(y_test.tolist(), estimated_values_test)
g.test_prediction(y_test.tolist(), estimated_values_test, title='True vs Estimated values comparison')

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
    
    regressor.train_pca(x_train.values, y_train.values, L=10)
    estimated_values = regressor.estimate_values(x_test, regressor.pca_parameters)
    
    r2 = r2_score(y_test.tolist(), estimated_values) 
    error = mean_squared_error(y_test.tolist(), estimated_values)
    r2_list.append(r2)
    mean_error.append(error)
    print r2, error    


print 'Mean R2 Score: ' + str(sum(r2_list)/len(r2_list))
print 'Mean Error: ' + str(sum(mean_error)/len(mean_error))


#### PLOT EIGVALUES
#
#N = float(len(x_train))
#R = covariance_matrix(x_train, N)
#eigvalues, eigvect = np.linalg.eig(R)
#
#fontlabel = 20
#matplotlib.rcParams['xtick.labelsize'] = fontlabel 
#matplotlib.rcParams['ytick.labelsize'] = fontlabel 
#matplotlib.rc('axes', edgecolor='black')
#fig, ax = plt.subplots()
#ax.plot(range(18), eigvalues/eigvalues.sum(), color='black')
#ax.fill_between(range(18), 0, eigvalues/eigvalues.sum(), facecolor='gray', alpha=0.5)
#ax.set_xlim(0, 17)
#ax.set_xlabel('#feature', fontsize=20)
#ax.set_ylabel('information', fontsize=20)
#ax.patch.set_facecolor('white')
#ax.grid(c='gray')