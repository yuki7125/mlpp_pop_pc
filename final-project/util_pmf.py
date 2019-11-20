#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
import pickle


def data_prepare_load(checker=1):
    #  When you HAVEN'T preprocessed the data yet
    if checker == 0:
        # Data from:
        # https://www.kaggle.com/rounakbanik/the-movies-dataset/download
        ratings_df = pd.read_csv(
            '../data/the-movies-dataset/ratings_small.csv')

        # We will only use part of the data that has more than 4 reviews per
        # movie
        moviecounts = ratings_df.movieId.value_counts()
        ratings_df = ratings_df[ratings_df.movieId.isin(
            moviecounts.index[moviecounts.gt(4)])].reset_index(drop=True)

        # Check the minimum number of ratings a user has posted
        # print('Minimum number of ratings for each user is '
        # + str(ratings_df.groupby('userId').size().min()))
        # print('Minimum number of ratings for each movie is '
        # + str(ratings_df.groupby('movieId').size().min()))

        ratings_df.to_csv('../data/ratings_preprocessed.csv')

    else:
        ratings_df = pd.read_csv(
            '../data/ratings_preprocessed.csv', index_col=0)

    return ratings_df


def train_test_split(ratings_df, checker=1):
    # Randomly split the data into train and test
    # Specifically, take 5 random ratings out of each user's rating list
    # (therefore, there will be 5 times # of user ratings in the test set)
    # For each user id, take out 5 ratings
    # out of the dataframe then append them into a new dataframe.
    # Then take the difference of the two data frames and the difference will
    # be the train data

    # Run this when you DON'T have a train_ratings.csv
    if checker == 0:
        for userid in ratings_df.userId.unique():
            if userid == 1:
                test_ratings = ratings_df[
                    ratings_df['userId'] == userid].sample(
                    5, random_state=0)
            else:
                test_ratings = test_ratings.append(
                    ratings_df[ratings_df['userId'] == userid].sample(
                        5, random_state=0))
        train_ratings = pd.concat([ratings_df, test_ratings]).drop_duplicates(
            keep=False).reset_index(drop=True)
        test_ratings = test_ratings.reset_index(drop=True)
        train_ratings.to_csv('../data/train_ratings.csv')
        test_ratings.to_csv('../data/test_ratings.csv')
        print("Number of ratings in entire dataset is " + str(len(ratings_df)))
        print("Number of ratings in train dataset is " + str(
            len(train_ratings)))
        print("Number of ratings in test dataset is " + str(len(test_ratings)))
    else:
        train_ratings = pd.read_csv('../data/train_ratings.csv', index_col=0)
        test_ratings = pd.read_csv('../data/test_ratings.csv', index_col=0)
    return train_ratings, test_ratings


def reindex_train(train_ratings, checker=1):
    # Reindexing of userId and movieId.
    # Since the IDs of the users and movies have gaps in between, we reindex
    # the IDs so that it will be easier to manipulate the user x movie ratings
    # matrix.

    # When you DON'T have a indexed_train_ratings.csv
    if checker == 0:
        unique_userId = train_ratings.userId.unique()
        unique_movieId = train_ratings.movieId.unique()

        train_ratings['new_user_index'],
        train_ratings['new_movie_index'] = 0, 0

        for old_id, new_id in zip(unique_userId, range(len(unique_userId))):
            train_ratings['new_user_index'].iloc[
                train_ratings[
                    train_ratings['userId'] == old_id].index.tolist()] = new_id

        for old_id, new_id in zip(unique_movieId, range(len(unique_movieId))):
            train_ratings[
                'new_movie_index'].iloc[
                train_ratings[train_ratings[
                    'movieId'] == old_id].index.tolist()] = new_id

        train_ratings.to_csv('../data/indexed_train_ratings.csv')

    else:
        train_ratings = pd.read_csv(
            '../data/indexed_train_ratings.csv', index_col=0)

    return train_ratings


def zero_imputation(ratings_df, train_ratings, checker=1):
    # Run this when you DON'T have a zero_imputated_ratings.npy
    if checker == 0:
        zero_imputated_ratings = np.empty(
            (ratings_df.userId.nunique(), ratings_df.movieId.nunique()))

        for user in range(ratings_df.userId.nunique()):
            zero_imputated_ratings[user] = 0
            for column in train_ratings[train_ratings.new_user_index ==
                                        user]['new_movie_index']:
                zero_imputated_ratings[user, column] = train_ratings[(
                    train_ratings.new_user_index == user) & (
                    train_ratings.new_movie_index == column)]['rating']

        np.save('../data/zero_imputated_ratings.npy', zero_imputated_ratings)

    else:
        zero_imputated_ratings = np.load('../data/zero_imputated_ratings.npy')

    return zero_imputated_ratings


def calc_average_ratings(posterior_samples_u, posterior_samples_i):
    # Change calculation depending on dimensions
    if posterior_samples_u.shape[2] == 10:
        tensor = torch.mm(
            posterior_samples_u[0, :, :], posterior_samples_i[0, :, :].T)
        for i in range(1, posterior_samples_u.shape[0] - 1):
            tensor += torch.mm(posterior_samples_u[i, :, :],
                               posterior_samples_i[i, :, :].T)

    elif posterior_samples_u.shape[1] == 10:
        tensor = torch.mm(
            posterior_samples_u[0, :, :].T, posterior_samples_i[0, :, :])
        for i in range(1, posterior_samples_u.shape[0] - 1):
            tensor += torch.mm(posterior_samples_u[i,
                                                   :, :].T,
                               posterior_samples_i[i, :, :])

    tensor = tensor / posterior_samples_u.shape[0]
    rounded_estimate_ratings = (tensor * 2).round() / 2
    rounded_estimate_ratings[rounded_estimate_ratings > 5.0] = 5.0
    rounded_estimate_ratings[rounded_estimate_ratings < 0.5] = 0.5

    return rounded_estimate_ratings


def plot_rounded_estimates(rounded_estimate_ratings):
    rounded_estimate_ratings_unique = rounded_estimate_ratings.unique(
        sorted=True)
    rounded_estimate_ratings_counts = torch.stack([
        (rounded_estimate_ratings == rounded_estimate).sum()
        for rounded_estimate in rounded_estimate_ratings_unique])
    plt.bar(
        rounded_estimate_ratings_unique,
        rounded_estimate_ratings_counts,
        width=0.2)


def model_assessment(train_ratings, test_ratings):
    # Discrepancies of original data (For PPC)
    train_mean_rating = train_ratings['rating'].mean()
    train_median_rating = train_ratings['rating'].median()
    train_first_q_rating = np.quantile(train_ratings['rating'], 0.25)
    train_third_q_rating = np.quantile(train_ratings['rating'], 0.75)

    # Discrepancies of new data (For POP-PC)
    test_mean_rating = test_ratings['rating'].mean()
    test_median_rating = test_ratings['rating'].median()
    test_first_q_rating = np.quantile(test_ratings['rating'], 0.25)
    test_third_q_rating = np.quantile(test_ratings['rating'], 0.75)

    with open('../data/pmf_rounded_posterior_samples.pickle', 'rb') as f:
        pmf_rounded_estimate_ratings = pickle.load(f)

    with open('../data/bpmf_rounded_posterior_samples.pickle', 'rb') as f:
        bpmf_rounded_estimate_ratings = pickle.load(f)

    with open('../data/poissonmf_rounded_posterior_samples.pickle', 'rb') as f:
        poi_rounded_estimate_ratings = pickle.load(f)

    # Discrepancies of predictions
    pmf_pred_mean_rating = torch.mean(pmf_rounded_estimate_ratings)
    pmf_pred_median_rating = torch.median(pmf_rounded_estimate_ratings)
    pmf_pred_first_q_rating = np.quantile(
        pmf_rounded_estimate_ratings.numpy(), 0.25)
    pmf_pred_third_q_rating = np.quantile(
        pmf_rounded_estimate_ratings.numpy(), 0.75)

    bpmf_pred_mean_rating = torch.mean(bpmf_rounded_estimate_ratings)
    bpmf_pred_median_rating = torch.median(bpmf_rounded_estimate_ratings)
    bpmf_pred_first_q_rating = np.quantile(
        bpmf_rounded_estimate_ratings.numpy(), 0.25)
    bpmf_pred_third_q_rating = np.quantile(
        bpmf_rounded_estimate_ratings.numpy(), 0.75)

    poi_pred_mean_rating = torch.mean(poi_rounded_estimate_ratings)
    poi_pred_median_rating = torch.median(poi_rounded_estimate_ratings)
    poi_pred_first_q_rating = np.quantile(
        poi_rounded_estimate_ratings.numpy(), 0.25)
    poi_pred_third_q_rating = np.quantile(
        poi_rounded_estimate_ratings.numpy(), 0.75)

    ppc_mean = [train_mean_rating - pmf_pred_mean_rating,
                train_mean_rating - bpmf_pred_mean_rating,
                train_mean_rating - poi_pred_mean_rating]
    ppc_median = [train_median_rating - pmf_pred_median_rating,
                  train_median_rating - bpmf_pred_median_rating,
                  train_median_rating - poi_pred_median_rating]
    ppc_first_q = [train_first_q_rating - pmf_pred_first_q_rating,
                   train_first_q_rating - bpmf_pred_first_q_rating,
                   train_first_q_rating - poi_pred_first_q_rating]
    ppc_third_q = [train_third_q_rating - pmf_pred_third_q_rating,
                   train_third_q_rating - bpmf_pred_third_q_rating,
                   train_third_q_rating - poi_pred_third_q_rating]

    poppc_mean = [test_mean_rating - pmf_pred_mean_rating,
                  test_mean_rating - bpmf_pred_mean_rating,
                  test_mean_rating - poi_pred_mean_rating]
    poppc_median = [test_median_rating - pmf_pred_median_rating,
                    test_median_rating - bpmf_pred_median_rating,
                    test_median_rating - poi_pred_median_rating]
    poppc_first_q = [test_first_q_rating - pmf_pred_first_q_rating,
                     test_first_q_rating - bpmf_pred_first_q_rating,
                     test_first_q_rating - poi_pred_first_q_rating]
    poppc_third_q = [test_third_q_rating - pmf_pred_third_q_rating,
                     test_third_q_rating - bpmf_pred_third_q_rating,
                     test_third_q_rating - poi_pred_third_q_rating]
    return ppc_mean, ppc_median, ppc_first_q, ppc_third_q, \
        poppc_mean, poppc_median, poppc_first_q, poppc_third_q


def plot_poppc_ppc(ppc_mean, ppc_median,
                   ppc_first_q, ppc_third_q,
                   poppc_mean, poppc_median,
                   poppc_first_q, poppc_third_q):
    labels = ['pmf', 'bpmf', 'poi']
    plt.subplot(2, 2, 1)
    plt.plot(labels, ppc_mean, labels, poppc_mean)
    plt.legend(['ppc', 'poppc'])
    plt.title('Mean Discrepancy')
    plt.subplot(2, 2, 2)
    plt.plot(labels, ppc_median, labels, poppc_median)
    plt.legend(['ppc', 'poppc'])
    plt.title('Median Discrepancy')
    plt.subplot(2, 2, 3)
    plt.plot(labels, ppc_first_q, labels, poppc_first_q)
    plt.legend(['ppc', 'poppc'])
    plt.title('First_q Discrepancy')
    plt.subplot(2, 2, 4)
    plt.plot(labels, ppc_third_q, labels, poppc_third_q)
    plt.legend(['ppc', 'poppc'])
    plt.title('Third_q Discrepancy')
    plt.tight_layout()
