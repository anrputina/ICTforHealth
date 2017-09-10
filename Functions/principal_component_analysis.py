#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 17 16:46:54 2017

@author: anr.putina
"""
import numpy as np
from data_manipulation import covariance_matrix

class PrincipalComponentAnalysis():
    
    def __init__(self):
        pass
    
    def z_matrix(self, x, L=0, p=0.999):
    
        N = float(len(x))
        R = covariance_matrix(x, N)
    
        eigvalues, eigvect = np.linalg.eig(R)
        
        if (L > 0):
            L = len(np.where(eigvalues.cumsum() < eigvalues.sum() * p)[0])   
            eigvect = eigvect[:, :L]
    
        z = x.dot(eigvect)
        z = z/z.std()
    
        return z