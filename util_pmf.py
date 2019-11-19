#!/usr/bin/env python
# coding: utf-8

import time

import pandas as pd
import numpy as np
import torch
import pyro
import pyro.distributions as dist
import matplotlib.pyplot as plt
import pickle 

from pyro import poutine
from numpy.linalg import inv
from pyro.infer.mcmc.api import MCMC
from pyro.infer.mcmc import NUTS


def data_prepare_load(checker=1):
    #  When you HAVEN'T preprocessed the data yet
    if checker == 0:
        # Data from: https://www.kaggle.com/rounakbanik/the-movies-dataset/download
        ratings_df = pd.read_csv('../data/the-movies-dataset/ratings_small.csv')
        
        # We will only use part of the data that has more than 4 reviews per movie
        moviecounts = ratings_df.movieId.value_counts()
        ratings_df = ratings_df[ratings_df.movieId.isin(moviecounts.index[moviecounts.gt(4)])].reset_index(drop=True)
        
        # Check the minimum number of ratings a user has posted
        # print('Minimum number of ratings for each user is ' + str(ratings_df.groupby('userId').size().min()))
        # print('Minimum number of ratings for each movie is ' + str(ratings_df.groupby('movieId').size().min()))
        
        ratings_df.to_csv('../data/ratings_preprocessed.csv')
        
    else:
        ratings_df = pd.read_csv('../data/ratings_preprocessed.csv', index_col=0)
    
    return ratings_df
     
    
def train_test_split(ratings_df, checker=1):
    # Randomly split the data into train and test
    # Specifically, take 5 random ratings out of each user's rating list (therefore, there will be 5 times # of user ratings in the test set)
    # For each user id, take out 5 ratings oaut of the dataframe then append them into a new dataframe.
    # Then take the difference of the two data frames and the difference will be the train data  
    
    # Run this when you DON'T have a train_ratings.csv
    if checker == 0:
        for userid in ratings_df.userId.unique() :
            if userid == 1:
                test_ratings = ratings_df[ratings_df['userId']==userid].sample(5, random_state=0)
            else:
                test_ratings = test_ratings.append(ratings_df[ratings_df['userId']==userid].sample(5, random_state=0))
        train_ratings = pd.concat([ratings_df, test_ratings]).drop_duplicates(keep=False).reset_index(drop=True)
        test_ratings = test_ratings.reset_index(drop=True)
        train_ratings.to_csv('../data/train_ratings.csv')
        test_ratings.to_csv('../data/test_ratings.csv')
        print("Number of ratings in entire dataset is "+ str(len(ratings_df)))
        print("Number of ratings in train dataset is "+  str(len(train_ratings)))
        print("Number of ratings in test dataset is "+ str(len(test_ratings)))
    else:
        train_ratings = pd.read_csv('../data/train_ratings.csv', index_col=0)
        test_ratings = pd.read_csv('../data/test_ratings.csv', index_col=0)
    return train_ratings, test_ratings


def reindex_train(train_ratings, checker=1):
    # Reindexing of userId and movieId.
    # Since the IDs of the users and movies have gaps in between, we reindex the IDs so that it will be easier to manipulate the user x movie ratings matrix.
    
    # When you DON'T have a indexed_train_ratings.csv 
    if checker == 0:
        unique_userId = train_ratings.userId.unique()
        unique_movieId = train_ratings.movieId.unique()

        train_ratings['new_user_index'], train_ratings['new_movie_index'] = 0, 0

        for old_id, new_id in zip(unique_userId, range(len(unique_userId))):
            train_ratings['new_user_index'].iloc[train_ratings[train_ratings['userId']==old_id].index.tolist()] = new_id

        for old_id, new_id in zip(unique_movieId, range(len(unique_movieId))):
            train_ratings['new_movie_index'].iloc[train_ratings[train_ratings['movieId']==old_id].index.tolist()] = new_id

        train_ratings.to_csv('../data/indexed_train_ratings.csv')
    
    else:
        train_ratings = pd.read_csv('../data/indexed_train_ratings.csv', index_col=0)
    
    return train_ratings


def zero_imputation(ratings_df, train_ratings, checker=1):
    # Run this when you DON'T have a zero_imputated_ratings.npy
    if checker == 0:
        zero_imputated_ratings = np.empty((ratings_df.userId.nunique(),ratings_df.movieId.nunique()))

        for user in range(ratings_df.userId.nunique()):
            zero_imputated_ratings[user] = 0
            for column in train_ratings[train_ratings.new_user_index == user]['new_movie_index']:
                zero_imputated_ratings[user, column] = train_ratings[(train_ratings.new_user_index == user)&(train_ratings.new_movie_index == column)]['rating']  

        np.save('../data/zero_imputated_ratings.npy', zero_imputated_ratings)
        
    else:
        zero_imputated_ratings = np.load('../data/zero_imputated_ratings.npy')
        
    return zero_imputated_ratings


def calc_average_ratings(posterior_samples_u, posterior_samples_i):
    # Change calculation depending on dimensions
    if posterior_samples_u.shape[2] == 10:
        tensor =  torch.mm(posterior_samples_u[0,:,:], posterior_samples_i[0,:,:].T)
        for i in range(1, posterior_samples_u.shape[0]-1):
            tensor += torch.mm(posterior_samples_u[i,:,:], posterior_samples_i[i,:,:].T)
       
    elif posterior_samples_u.shape[1] == 10:
        tensor =  torch.mm(posterior_samples_u[0,:,:].T, posterior_samples_i[0,:,:])
        for i in range(1, posterior_samples_u.shape[0]-1):
            tensor += torch.mm(posterior_samples_u[i,:,:].T, posterior_samples_i[i,:,:])
        
    tensor = tensor/posterior_samples_u.shape[0]
    rounded_estimate_ratings = (tensor* 2).round()/ 2
    rounded_estimate_ratings[rounded_estimate_ratings>5.0] = 5.0
    rounded_estimate_ratings[rounded_estimate_ratings<0.5] = 0.5
    
    return rounded_estimate_ratings

def plot_rounded_estimates(rounded_estimate_ratings):
    rounded_estimate_ratings_unique = rounded_estimate_ratings.unique(sorted=True)
    rounded_estimate_ratings_counts = torch.stack([(rounded_estimate_ratings==rounded_estimate).sum() for rounded_estimate in rounded_estimate_ratings_unique])
    plt.bar(rounded_estimate_ratings_unique, rounded_estimate_ratings_counts, width=0.2)
    
