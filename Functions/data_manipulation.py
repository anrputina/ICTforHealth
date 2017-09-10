#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 10 22:29:01 2017

@author: anr.putina
"""

import pandas as pd
import numpy as np

def load_data(name):
    df = pd.read_csv('../Data/'+name)
    df = prepare_data(df)
    return df

def load_arrhythmia_data(name, clusters='full'):
    
    df = pd.read_csv('../Data/'+name, header=None)
    
    if clusters=='boolean':
        df[df.columns.size-1] = df[df.columns.size-1].apply(lambda x: 2 if x >= 2 else 1)

    df = df.rename(columns={df.columns.size-1: 'class'})
    
    df = df.replace({"?": np.NaN}).dropna(axis=1)

    df = df[df.columns[(df != 0).any()]]
    
    return df

def prepare_data(df):
    df.test_time = df.test_time.abs().round()
    return df

def remove_columns(df, *args):
    
    for feature in args:
        df = df.drop(feature, axis=1)
    return df
    
def select_training_df(df, start, end):
    training_df = df[(df['subject#'] >= start)& \
                     (df['subject#'] <= end)]
    return training_df

def select_training_df_bySubject(df, lista):
    df_ = df[df['subject#'].isin(lista)]
    
    return df_

def select_testing_df(df, start, end):
    testing_df = df[(df['subject#'] >= start)& \
                     (df['subject#'] <= end)]
    return testing_df

def normalize_matrix(df):
    return (df - df.mean())/df.std()

def select_y_byID(df, feature):
    return df[df.columns[feature]]

def select_y_byName(df, feature):
    return df[feature]

def select_x_byID(df, feature):
    return df.drop(df.columns[feature], axis=1)

def select_x_byName(df, feature):
    return df.drop(feature, axis=1)

def covariance_matrix(x_train, N):
    return 1.0/N * x_train.transpose().dot(x_train)