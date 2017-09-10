#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 17 16:54:40 2017

@author: anr.putina
"""

import numpy as np
import pandas as pd
import random as rd
from principal_component_analysis import PrincipalComponentAnalysis
from data_manipulation import covariance_matrix

class Classifier():
    
    def __init__(self):
        pass
    
    def min_dist_classification(self, df, class_id, classes, out='prediction'):
                
        xmeans = np.empty(shape=[len(classes), df.shape[1]])
    
        for cl in classes:
            mean = df[class_id==cl].mean()
            xmeans[cl-1] = mean
                        
        #### V1 ###
        eny = np.diag((df.values).dot(df.values.T))
        enx = np.diag(xmeans.dot(xmeans.T))
        dotprod = df.values.dot(xmeans.T)
        xv, yv = np.meshgrid(enx, eny)
        dist = xv + yv - 2*dotprod
        dist = pd.DataFrame(dist).dropna(axis=1)
        
        #### V2 ###
#        def square_distance(x, xk):
#            return np.linalg.norm(x-xk)**2
#        
#        dist = pd.DataFrame()
#        dist[0] = df.apply(lambda x: (square_distance(x, xmeans[0])), axis=1)
#        dist[1] = df.apply(lambda x: (square_distance(x, xmeans[1])), axis=1)
        
        ####
        if out == 'prediction':    
            df['previsione'] = pd.DataFrame(np.argmin(dist.values, axis=1)).apply(lambda x: x+1)
            return df
        
        elif out=='dist':
            return dist
        
    def pca_classification(self, df, class_id, classes, classification = 'min_dist'):
        pca = PrincipalComponentAnalysis()
        
        z_matrix = pca.z_matrix(df, L=10, p=0.999)
        distances = self.min_dist_classification(z_matrix, class_id, classes, out='dist')
        
        if classification == 'distance_bayes':
            
            pi = [0 for x in range(1,17)]
            for cl in classes:
                pi[cl-1] = float((class_id == cl).sum())/float(len(class_id))    
            for cl in classes:
                if (pi[cl-1] > 0):
                    distances[cl-1] = distances[cl-1].apply(lambda x: x - 2 * np.log(pi[cl-1]))

        elif classification == 'region_bayes':
            
            distances = pd.DataFrame()

            for cl in classes:
                group_mean = z_matrix[class_id==cl].mean()
                group = z_matrix[class_id==cl] - group_mean
                
                if len(group) > 0 :
                    Rx = covariance_matrix(group, float(len(group)))
                    pi = float((class_id == cl).sum())/float(len(class_id))
                    
                    Rxi = np.linalg.inv(Rx)
                    Rxdet = np.linalg.det(Rx)
                    print Rxdet
                    
                    distances[cl] = z_matrix.apply(lambda x: (x-group_mean).dot(Rxi).dot(x-group_mean).T+\
                                                              np.log(Rxdet)-\
                                                              2 * np.log(pi),\
                                                              axis=1)
                else:
                    pass
                
        df['previsione'] = pd.DataFrame(np.argmin(distances.values, axis=1)).apply(lambda x: x+1)

        return df

    def kmeans(self, df, k, class_id=None, start='random', var=None):
        
        iterations = 0
        
        if ((start == 'mean') & (class_id is not None)):
            class_id_kmeans = class_id
        else:
            class_id_kmeans =  pd.Series([rd.randint(1,2) for x in range(len(df))])

        while True:
    
            dist = self.min_dist_classification(df, class_id_kmeans, range(1,k+1), out='dist')
            
#            if var =='update':
                
                
            
            classification = pd.Series(np.argmin(dist.values, axis=1)+1)
        
            if  (classification-class_id_kmeans).abs().sum() == 0:
                break
            
            class_id_kmeans = classification
            iterations += 1
            
        print 'Convergence reached in {}'.format(iterations)
        return classification
            
        
        
    def check_detections(self, estimated_values, true_values):
        
        true_positive = float(((estimated_values >= 2) & (true_values >= 2)).sum())/\
                        float((true_values >= 2).sum())
        true_negative = float(((estimated_values < 2) & (true_values < 2)).sum())/\
                        float((true_values < 2).sum())
        false_positive = float(((estimated_values >= 2) & (true_values < 2)).sum())/\
                         float((true_values < 2).sum())
        false_negative = float(((estimated_values < 2) & (true_values >= 2)).sum())/\
                         float((true_values >= 2).sum())
        
        print 'True positive: {}'.format(true_positive)
        print 'True negative: {}'.format(true_negative)
        print 'False positive: {}'.format(false_positive)
        print 'False negative: {}'.format(false_negative)
    
        print 'Precision: {}'.format(true_positive/(true_positive+false_positive))
        print 'Recall: {}'.format(true_positive/(true_positive+false_negative))
        print 'Specificity: {}'.format(true_negative/(true_negative+false_positive))
        
    def get_detections(self, estimated_values, true_values):
        
        true_positive = float(((estimated_values >= 2) & (true_values >= 2)).sum())/\
                        float((true_values >= 2).sum())
        true_negative = float(((estimated_values < 2) & (true_values < 2)).sum())/\
                        float((true_values < 2).sum())
        false_positive = float(((estimated_values >= 2) & (true_values < 2)).sum())/\
                         float((true_values < 2).sum())
        false_negative = float(((estimated_values < 2) & (true_values >= 2)).sum())/\
                         float((true_values >= 2).sum())
        
        precision = true_positive/(true_positive+false_positive)
        recall = true_positive/(true_positive+false_negative)
        specificity = true_negative/(true_negative+false_positive)
        
        return recall, precision, specificity
        
    def check_detections_NN(self, estimated_values, true_values):
        
        true_positive = float(((estimated_values >= 1) & (true_values >= 1)).sum())/\
                        float((true_values >= 1).sum())
        true_negative = float(((estimated_values < 1) & (true_values < 1)).sum())/\
                        float((true_values < 1).sum())
        false_positive = float(((estimated_values >= 1) & (true_values < 1)).sum())/\
                         float((true_values < 1).sum())
        false_negative = float(((estimated_values < 1) & (true_values >= 1)).sum())/\
                         float((true_values >= 1).sum())
        
        print 'True positive: {}'.format(true_positive)
        print 'True negative: {}'.format(true_negative)
        print 'False positive: {}'.format(false_positive)
        print 'False negative: {}'.format(false_negative)
    
        print 'Precision: {}'.format(true_positive/(true_positive+false_positive))
        print 'Recall: {}'.format(true_positive/(true_positive+false_negative))
        print 'Specificity: {}'.format(true_negative/(true_negative+false_positive))   

    def get_detections_NN(self, estimated_values, true_values):
        
        true_positive = float(((estimated_values >= 1) & (true_values >= 1)).sum())/\
                        float((true_values >= 1).sum())
        true_negative = float(((estimated_values < 1) & (true_values < 1)).sum())/\
                        float((true_values < 1).sum())
        false_positive = float(((estimated_values >= 1) & (true_values < 1)).sum())/\
                         float((true_values < 1).sum())
        false_negative = float(((estimated_values < 1) & (true_values >= 1)).sum())/\
                         float((true_values >= 1).sum())
                         
        precision = true_positive/(true_positive+false_positive)
        recall = true_positive/(true_positive+false_negative)
        specificity = true_negative/(true_negative+false_positive)
        
        return recall, precision, specificity