#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 12 14:36:49 2017

@author: anr.putina
"""

import matplotlib
import matplotlib.pyplot as plt
matplotlib.style.use('ggplot')

class Graphics():
    
    def __init__(self):
        self.figsize = (13,6)
        self.bins = 100
        self.linewidth = 0.2
        self.scatter_size = 10
    
    def test_prediction(self, true_values, estimated_values, title='Default'):
        
        plt.figure(figsize=self.figsize)
        plt.plot(true_values, 'r', linewidth=self.linewidth, label='True value')
        plt.plot(estimated_values, 'b--', linewidth=self.linewidth, label='Estimated Values')
        plt.xlabel('Sample [k]')
        plt.ylabel('Value')
        plt.title(title)
        plt.legend()
        
    def hist_errors(self, true_values, estimated_values, title='Error Distribution'):
        
        plt.figure(figsize=self.figsize)
        plt.hist(estimated_values - true_values, bins=self.bins, normed=True)
        plt.xlabel('Error')
        plt.ylabel('Probability p')
        plt.title(title)
        plt.legend()
        
    def hist_errors_comparison(self, true_train, estimated_train, true_test, estimated_test, title='Error Distribution'):

        fig, ax = plt.subplots(figsize=self.figsize)
        ax.hist(true_train - estimated_train, bins=self.bins, normed=True, label='Train', color='b', alpha=0.5, histtype='stepfilled')
        ax.hist(true_test - estimated_test, bins=self.bins, normed=True, label='Test',color='g', alpha=0.5, histtype='stepfilled')
        ax.legend()
        

    def test_pred(self, true_values, estimated_values, title='Boh'):
        
        plt.figure(figsize=self.figsize)
        plt.scatter(true_values, estimated_values, s=self.scatter_size)
        plt.plot(true_values, true_values)
        plt.title(title)

    def show_features(self, feature_1, feature_2, title='Feature comparison'):
        
        plt.figure(figsize=self.figsize)
        plt.scatter(feature_1, feature_2, s=self.scatter_size)
        plt.xlabel(feature_1.name)
        plt.ylabel(feature_2.name)
        plt.title(title)