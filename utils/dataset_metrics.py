import pandas as pd
import statistics
import numpy as np
import matplotlib.pyplot as plt
import pickle
import utils.keywords as kw


def convert_interactions(dict_interactions, dict_values, interactions_df):
    """Convert original interactions to our feedback componen and order it by timestamp."""
    interactions_df.eventType = interactions_df.eventType.apply(
        lambda x: dict_interactions[x])
    interactions_df['rating'] = interactions_df['eventType'].apply(
        lambda x: dict_values[x])
    # Order sessions by timestamp and  subset function
    interactions_ordered = interactions_df.sort_values(
        by='timestamp', ascending=True)
    interactions_ordered['contentId'] = interactions_ordered['contentId'].astype(
        str)
    interactions_ordered['personId'] = interactions_ordered['personId'].astype(
        str)
    return interactions_ordered


def get_subset(df_full_ordered, number):
    """Split dataset given a number of interactions."""
    interactions = df_full_ordered.iloc[0:number, :].groupby(
        ['personId', 'contentId'])['rating'].mean().reset_index()
    interactions.rating = interactions.rating.round()
    um = interactions.pivot(
        index='personId', columns='contentId', values='rating').fillna(0)
    return um, interactions


def get_overall_sparsity(ut_matrix):
    """Calculating overall sparsity for a given matrix."""
    A = np.array(ut_matrix)
    sparsity = round(1.0 - (np.count_nonzero(A) / float(A.size)), 4)
    return sparsity


def plot_user_activity_m1(matrix, USER):
    users = list(matrix.index)
    ratings_users = []
    for user in users:
        user_ratings = matrix.shape[1] - \
            (matrix == 0).astype(int).sum(axis=1)[user]
        ratings_users.append(user_ratings)
    x = range(len(ratings_users))
    y = ratings_users
    index_user = users.index(USER)
    plt.scatter(x, y, alpha=0.5)
    plt.scatter(index_user, ratings_users[index_user], color="yellow")
    plt.title('NUMBER OF RATINGS MADE BY USER')
    plt.xlabel('Users index')
    plt.ylabel('Number ratings')
    plt.show()
    dev_users = statistics.stdev(ratings_users)
    med_users = statistics.median(ratings_users)
    mean_users = statistics.mean(ratings_users)
    print("DEVIATION = {} \nMEDIAN = {} \nMEAN = {}".format(
        dev_users, med_users, mean_users))


def get_metric1(userid, ut_matrix):
    "Number of items that the user has rated devided by the maximum number of items rated by an user"
    "Comparison between my activity and the most active user"
    "Number of users remain almost the same - comparable in the same time"
    user_numb_ratings = ut_matrix.shape[1] - \
        (ut_matrix == 0).astype(int).sum(axis=1)[userid]
    max_ratings_by_an_user = ut_matrix.shape[1] - \
        min((ut_matrix == 0).astype(int).sum(axis=1))
    return round(user_numb_ratings/max_ratings_by_an_user, 3)


def plot_items_activity_m3(matrix):
    items = list(matrix.columns)
    ratings_items = []
    for item in items:
        item_ratings = matrix.shape[0] - \
            (matrix == 0).astype(int).sum(axis=0)[item]
        ratings_items.append(item_ratings)
    x = range(len(items))
    y = ratings_items
    plt.scatter(x, y, alpha=0.5)
    plt.title('NUMBER OF RATINGS FOR AN ITEM')
    plt.xlabel('Items index')
    plt.ylabel('Number ratings')
    plt.show()
    dev_items = statistics.stdev(ratings_items)
    med_items = statistics.median(ratings_items)
    mean_items = statistics.mean(ratings_items)
    print("DEVIATION = {} \nMEDIAN = {} \nMEAN = {}".format(
        dev_items, med_items, mean_items))


def get_metric3(ut_matrix):
    "Proportion of items rated by more than 5% users."
    items = list(ut_matrix.columns)
    list_ratings_items = []
    for item in items:
        item_ratings = ut_matrix.shape[0] - \
            (ut_matrix == 0).astype(int).sum(axis=0)[item]
        list_ratings_items.append(item_ratings)
    number_users = ut_matrix.shape[0]
    N = 0.05*number_users
    more_than_N = [it for it in list_ratings_items if it >= N]
    return round(len(more_than_N)/len(list_ratings_items), 3)


def plot_quality_descriptors_m4(utmatrix, tfidf_matrix, tfidf_feature_names, full_item_ids):
    items = list(utmatrix.columns)
    topn = 10
    qualities = []
    for it in items:
        qualities.append(kw.get_keywords(
            full_item_ids, tfidf_matrix, tfidf_feature_names, topn, it)[1])
    x = range(len(items))
    y = qualities
    plt.scatter(x, y, alpha=0.5)
    plt.title('M4 - QUALITY OF DESCRIPTION')
    plt.xlabel('Items index')
    plt.ylabel('Mean TDIDF value of the keywords')
    plt.show()
    dev_items = statistics.stdev(qualities)
    med_items = statistics.median(qualities)
    mean_items = statistics.mean(qualities)
    print("DEVIATION = {} \nMEDIAN = {} \nMEAN = {}".format(
        dev_items, med_items, mean_items))
    return mean_items
