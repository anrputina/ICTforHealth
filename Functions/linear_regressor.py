#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 10 23:13:05 2017

@author: anr.putina
"""

import numpy as np
from numpy.linalg import pinv
import pandas as pd
from data_manipulation import *
np.random.seed(10007)


def gradient(X, y, a):        
    return -2 * X.T.dot(y) + 2 * X.T.dot(X).dot(a)

class LinearRegressor():
    
    def __init__(self):
        pass
    
    def train_mse(self, x_train, y_train):
        
        matrix = x_train.as_matrix()
        self.mse_parameters = np.dot(np.dot(pinv(np.dot(matrix.transpose(),matrix)), matrix.transpose()), y_train)
    
    def train_gradient(self, x_train, y_train, gamma = 1.0e-5, epsilon=1.0e-6):
                
        a_hat = np.random.rand(x_train.shape[1])
        a_hat_ii = np.ones(x_train.shape[1])
        
        grad = gradient(x_train, y_train, a_hat)
        counter = 0
        
        while(np.linalg.norm(a_hat - a_hat_ii) > epsilon and counter < 100000):
            
            a_hat_ii = a_hat
            a_hat = a_hat - gamma * grad
            counter = counter + 1
            grad = gradient(x_train, y_train, a_hat)

        self.gradient_parameters = a_hat
        print 'Convergence Reached in {} iterations'.format(counter)    

    def train_stepeest_descent(self, x_train, y_train, epsilon=1.0e-6):
        
        def hessian (X):
            return 4 * X.T.dot(X)
        
        a_hat = np.random.rand(x_train.shape[1])
        a_hat_ii = np.ones(x_train.shape[1])
        grad = gradient(x_train, y_train, a_hat)
        hes = hessian(x_train)
        counter = 0
        
        while (np.linalg.norm(a_hat - a_hat_ii) > epsilon and counter < 100000):
            
            a_hat_ii = a_hat
            a_hat = a_hat - (np.linalg.norm(grad)**2)/(grad.transpose()\
                             .dot(hes).dot(grad)) * grad
            counter = counter + 1
            grad = gradient(x_train, y_train, a_hat)

        self.stepeest_descent_parameters = a_hat
        print 'Convergence Reached in {} iterations'.format(counter)    

    def train_pca(self, x_train, y_train, L=0):
        
        N = float(len(x_train))
        R = covariance_matrix(x_train, N)
    
        eigvalues, eigvect = np.linalg.eig(R)
        eigvalues = eigvalues * np.identity(len(eigvalues))
        
        if (L > 0):
            print eigvalues[:L].sum()
            print eigvalues.sum()
            print eigvalues[:L].sum() / eigvalues.sum()
            eigvect = eigvect[:, :L]
            eigvalues = eigvalues[:L, :L]

        
        self.pca_parameters = 1/N * eigvect.dot(np.linalg.inv(eigvalues))\
            .dot(eigvect.transpose()).dot(x_train.transpose()).dot(y_train)

    def Z_matrix(self, x, L=0, p=0.999):
        
        N = float(len(x))
        R = covariance_matrix(x, N)

        eigvalues, eigvect = np.linalg.eig(R)

        if (L > 0):
            L = len(np.where(eigvalues.cumsum() < eigvalues.sum() * p)[0])   
            eigvect = eigvect[:, :L]

        Z = x.dot(eigvect)
        Z = Z/Z.std()

        return Z

    def estimate_values(self, in_data, parameters):
        
        matrix = in_data.as_matrix()
        return np.dot(matrix, parameters)