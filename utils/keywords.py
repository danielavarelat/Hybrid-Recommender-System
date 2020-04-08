from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import scipy
import numpy as np
import pickle


def createTDIDF(articles_dataframe):
    """Creation of TDIDF matrix and names of features."""
    stopwords_list = stopwords.words('english') + stopwords.words('portuguese')
    vectorizer = TfidfVectorizer(analyzer='word',
                                 ngram_range=(1, 2),
                                 min_df=0.003,
                                 max_df=0.5,
                                 max_features=5000,
                                 stop_words=stopwords_list)
    full_tfidf_matrix = vectorizer.fit_transform(
        articles_dataframe['title'] + "" + articles_dataframe['text'])
    tfidf_names = vectorizer.get_feature_names()
    return full_tfidf_matrix, tfidf_names


def sort_coo(coo_matrix):
    tuples = zip(coo_matrix.col, coo_matrix.data)
    return sorted(tuples, key=lambda x: (x[1], x[0]), reverse=True)


def extract_topn_from_vector(feature_names, sorted_items, topn=10):
    """Get the feature names and tf-idf score of top n items."""
    sorted_items = sorted_items[:topn]
    score_vals = []
    feature_vals = []
    for idx, score in sorted_items:
        score_vals.append(round(score, 3))
        feature_vals.append(feature_names[idx])
    results = {}
    for idx in range(len(feature_vals)):
        results[feature_vals[idx]] = score_vals[idx]
    return results


def get_keywords(list_item_ids, tfidf_matrix, tfidf_feature_names, number, itemID):
    """Extract top n keywords for an item and the mean of tfidf values for the N keywords."""
    ind = list_item_ids.index(itemID)
    tf_idf_vector = tfidf_matrix[ind]
    sorted_items = sort_coo(tf_idf_vector.tocoo())
    keywords = extract_topn_from_vector(
        tfidf_feature_names, sorted_items, number)
    mean_keys = np.array(list(keywords.values())).mean()
    return keywords, round(mean_keys, 3)
