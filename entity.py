import pandas as pd
import numpy as np
import pickle
import scipy
import sklearn
import random
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import pairwise_distances


class User:
    def __init__(self, userid, uti_matrix, interactions_dataframe):
        """ Create User object."""
        self.id = userid
        self.utility_matrix = uti_matrix
        self.interactions_df = interactions_dataframe
        self.items_to_ignore = self.interactions_df[self.interactions_df['personId'] == self.id].contentId.unique(
        )


class CB:
    def __init__(self, user):
        """ Create User object."""
        self.user = user

    def get_items_profile(self, full_items_ids, items_ids, tfidf_matrix):
        item_profiles_list = [tfidf_matrix[full_items_ids.index(
            x):full_items_ids.index(x)+1] for x in items_ids]
        item_profiles = scipy.sparse.vstack(item_profiles_list)
        return item_profiles

    def build_user_profile(self, user_id, interactions, full_item_list, tfidf_matrix):
        interactions_person_df = interactions.set_index(
            'personId').loc[user_id]
        if isinstance(interactions_person_df.contentId, str):
            items = [interactions_person_df.contentId]
        else:
            items = list(interactions_person_df.contentId.unique())
        items_interacted_profiles = self.get_items_profile(
            full_item_list, items, tfidf_matrix)
        user_item_ratings = np.array(
            interactions_person_df['rating']).reshape(-1, 1)
        weighted_avg = np.sum(items_interacted_profiles.multiply(
            user_item_ratings), axis=0) / np.sum(user_item_ratings)
        user_profile_normalized = sklearn.preprocessing.normalize(weighted_avg)
        return user_profile_normalized

    def important_words(self, profile, n, tfidf_feature_names):
        # topn = sorted(profile[0, :], reverse=True)[0:n]
        index_topn = np.argsort(-profile[0, :])[0:n]
        main_words = [tfidf_feature_names[idx] for idx in index_topn]
        return main_words

    def compute_similarities(self, interactions_df, USER, full_item_ids, tfidf_matrix, items_to_ignore):
        items_subset = list(interactions_df.contentId.unique())
        items_to_compare = [
            x for x in items_subset if x not in self.user.items_to_ignore]
        subset_items_profiles = self.get_items_profile(
            full_item_ids, items_to_compare, tfidf_matrix)
        profile_user = self.build_user_profile(
            USER, interactions_df, full_item_ids, tfidf_matrix)
        cosine_similarities = cosine_similarity(
            profile_user, subset_items_profiles)
        results = pd.DataFrame(cosine_similarities,
                               columns=items_to_compare, index=[USER])
        return results


class MF:
    def __init__(self, user):
        """ Create User object."""
        self.user = user

    def predict_matrix(self, utility_matrix, factors):
        """Full matrix with values predicted after factorization."""
        users_items_pivot_matrix = utility_matrix.as_matrix()
        users_items_pivot_sparse_matrix = csr_matrix(users_items_pivot_matrix)
        users_ids = list(utility_matrix.index)
        U, sigma, Vt = svds(users_items_pivot_sparse_matrix, k=factors)
        sigma = np.diag(sigma)
        all_predicted_ratings = np.dot(np.dot(U, sigma), Vt)
        all_norm = (all_predicted_ratings - all_predicted_ratings.min()) / \
            (all_predicted_ratings.max() - all_predicted_ratings.min())
        cf_preds_df = pd.DataFrame(
            all_norm, columns=utility_matrix.columns, index=users_ids)
        return cf_preds_df

    def predictions_user(self, factors):
        """Remove items the user has already interacted with."""
        df = self.predict_matrix(self.user.utility_matrix, factors)
        all_items = list(self.user.utility_matrix.columns)
        results_user = df.loc[self.user.id, [
            it for it in all_items if it not in self.user.items_to_ignore]].to_frame().T
        return results_user


class CF:
    def __init__(self, user):
        """ Create User object."""
        self.user = user

    def getNneighbors(self, N, user_id, utility_matrix):
        """Calculate similarity between users and get N neihgbors."""
        index_user = list(utility_matrix.index).index(user_id)
        model_knn = NearestNeighbors(metric="cosine", algorithm='brute')
        model_knn.fit(utility_matrix)
        distances, indices = model_knn.kneighbors(
            utility_matrix.iloc[index_user, :].values.reshape(1, -1), n_neighbors=N+1)
        similarities = list(1-distances.flatten())
        similarities = similarities[1:]  # remove the same item
        indices = list(indices.flatten())
        indices.remove(index_user)
        users_neigs = [list(utility_matrix.index)[i] for i in indices]
        similarities_list = [similarities[i] for i in range(len(indices))]
        return users_neigs, similarities_list, indices

    def CalculatePrediction(self, N, user_id, itemid,  utility_matrix):
        index_user = list(utility_matrix.index).index(user_id)
        mean_rating_user = utility_matrix.iloc[index_user, :].mean()
        _, similarities, indices = self.getNneighbors(
            N, user_id, utility_matrix)
        sum_n_similarities = np.sum(similarities)
        wtd_sum = 0
        for index, i in enumerate(indices):
            user_neig = list(utility_matrix.index)[i]
            rating_neig = utility_matrix.loc[user_neig][itemid]
            average_neig = np.mean(utility_matrix.iloc[i, :])
            rating_diff = rating_neig - average_neig
            product = rating_diff * (similarities[index])
            wtd_sum = wtd_sum + product
        prediction = mean_rating_user + (wtd_sum/sum_n_similarities)
        return prediction

    def predict_ratings(self, N, user_id, utility_matrix, items_ignore):
        items_subset = list(utility_matrix.columns)
        items_to_use = [x for x in items_subset if x not in items_ignore]
        results = pd.DataFrame(columns=items_to_use, index=[user_id])
        for col in items_to_use:
            results[col] = self.CalculatePrediction(
                N, user_id, col, utility_matrix)
        return results
