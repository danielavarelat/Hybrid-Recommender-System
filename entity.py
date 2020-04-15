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
    def __init__(self, user, full_items_ids, tfidf_matrix):
        """ Create User object."""
        self.user = user
        self.all_items = full_items_ids
        self.tfidf_matrix = tfidf_matrix

    def get_items_profile(self, items_ids):
        item_profiles_list = [self.tfidf_matrix[self.all_items.index(
            x):self.all_items.index(x)+1] for x in items_ids]
        item_profiles = scipy.sparse.vstack(item_profiles_list)
        return item_profiles

    def build_user_profile(self):
        interactions_person_df = self.user.interactions_df.set_index(
            'personId').loc[self.user.id]  # FALLA SI NO HAY INTERACCIONES
        if isinstance(interactions_person_df.contentId, str):
            items = [interactions_person_df.contentId]
        else:
            items = list(interactions_person_df.contentId.unique())
        items_interacted_profiles = self.get_items_profile(items)
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

    def compute_similarities(self):
        items_subset = list(self.user.utility_matrix.columns)
        items_to_compare = [
            x for x in items_subset if x not in self.user.items_to_ignore]

        subset_items_profiles = self.get_items_profile(items_to_compare)
        profile_user = self.build_user_profile()
        cosine_similarities = cosine_similarity(
            profile_user, subset_items_profiles)
        results = pd.DataFrame(cosine_similarities,
                               columns=items_to_compare, index=[self.user.id])
        return results


class MF:
    def __init__(self, user, number_factors):
        """ Create User object."""
        self.user = user
        self.factors = number_factors

    def predict_matrix(self):
        """Full matrix with values predicted after factorization."""
        users_items_pivot_matrix = self.user.utility_matrix.as_matrix()
        users_items_pivot_sparse_matrix = csr_matrix(users_items_pivot_matrix)
        users_ids = list(self.user.utility_matrix.index)
        U, sigma, Vt = svds(users_items_pivot_sparse_matrix, k=self.factors)
        sigma = np.diag(sigma)
        all_predicted_ratings = np.dot(np.dot(U, sigma), Vt)
        all_norm = (all_predicted_ratings - all_predicted_ratings.min()) / \
            (all_predicted_ratings.max() - all_predicted_ratings.min())
        cf_preds_df = pd.DataFrame(
            all_norm, columns=self.user.utility_matrix.columns, index=users_ids)
        return cf_preds_df

    def predictions_user(self):
        """Remove items the user has already interacted with."""
        df = self.predict_matrix()
        all_items = list(self.user.utility_matrix.columns)
        results_user = df.loc[self.user.id, [
            it for it in all_items if it not in self.user.items_to_ignore]].to_frame().T
        return results_user


class CF:
    def __init__(self, user, N):
        """ Create User object."""
        self.user = user
        self.N_neighbors = N

    def getNneighbors(self):
        """Calculate similarity between users and get N neihgbors."""
        index_user = list(self.user.utility_matrix.index).index(self.user.id)
        model_knn = NearestNeighbors(metric="cosine", algorithm='brute')
        model_knn.fit(self.user.utility_matrix)
        distances, indices = model_knn.kneighbors(
            self.user.utility_matrix.iloc[index_user, :].values.reshape(1, -1), n_neighbors=self.N_neighbors+1)
        similarities = list(1-distances.flatten())
        similarities = similarities[1:]  # remove itself
        indices = list(indices.flatten())
        indices.remove(index_user)
        users_neigs = [list(self.user.utility_matrix.index)[i]
                       for i in indices]
        similarities_list = [similarities[i] for i in range(len(indices))]
        return users_neigs, similarities_list, indices

    def CalculatePrediction(self, itemid):
        index_user = list(self.user.utility_matrix.index).index(self.user.id)
        mean_rating_user = self.user.utility_matrix.iloc[index_user, :].mean()
        _, similarities, indices = self.getNneighbors()
        sum_n_similarities = np.sum(similarities)
        wtd_sum = 0
        for index, i in enumerate(indices):
            user_neig = list(self.user.utility_matrix.index)[i]
            rating_neig = self.user.utility_matrix.loc[user_neig][itemid]
            average_neig = np.mean(self.user.utility_matrix.iloc[i, :])
            rating_diff = rating_neig - average_neig
            product = rating_diff * (similarities[index])
            wtd_sum = wtd_sum + product
        prediction = mean_rating_user + (wtd_sum/sum_n_similarities)
        return prediction

    def predict_ratings(self):
        items_subset = list(self.user.utility_matrix.columns)
        items_to_use = [
            x for x in items_subset if x not in self.user.items_to_ignore]
        results = pd.DataFrame(columns=items_to_use, index=[self.user.id])
        for col in items_to_use:
            results[col] = self.CalculatePrediction(col)
        return results

    def default_rating(self):
        us, sim, ind = self.getNneighbors()
        sum_n_similarities = np.sum(sim)
        index_user = list(self.user.utility_matrix.index).index(self.user.id)
        mean_rating_user = self.user.utility_matrix.iloc[index_user, :].mean()
        wtd_sum = 0
        for index, i in enumerate(ind):
            product = - \
                np.mean(self.user.utility_matrix.iloc[i, :]) * (sim[index])
            wtd_sum = wtd_sum + product
        prediction = mean_rating_user + (wtd_sum/sum_n_similarities)
        interactions_neihgs = []
        for i in us:
            interactions_neihgs.append(list(
                self.user.interactions_df[self.user.interactions_df['personId'] == i].contentId.unique()))
        flat_list = [
            item for sublist in interactions_neihgs for item in sublist]
        items = [i for i in set(flat_list)
                 if i not in self.user.items_to_ignore]
        return prediction, items


class HYBRID():
    def __init__(self, user, number_factors_mf, number_neighbors_cf, full_items_ids, tfidf_matrix):
        """ Create User object."""
        self.user = user
        self.factors_mf = number_factors_mf
        self.N_cf = number_neighbors_cf
        self.all_items = full_items_ids
        self.tfidf_matrix = tfidf_matrix

    def get_all_predictions(self):
        """Dataframe with three algorithms prediction for an user."""
        cb = CB(self.user,  self.all_items, self.tfidf_matrix,)
        cb_predictions = cb.compute_similarities()
        mf = MF(self.user, self.factors_mf)
        mf_predictions = mf.predictions_user()
        cf = CF(self.user, self.N_cf)
        cf_predictions = cf.predict_ratings()
        df_results = cb_predictions.append(mf_predictions, sort=True)
        df_results = df_results.append(cf_predictions, sort=True)
        df_results.insert(0, 'algorithm', ['CB', 'MF', 'CF'])
        return df_results

    def get_hybrid_predictions(self, weights):
        """Dictionary of items and its hybrid prediction given the weights."""
        #weights = [wcb, wmf, wcf]
        dict_hybrid = {}
        df_all = self.get_all_predictions()
        df_all.pop('algorithm')
        for item in df_all:
            preds = df_all[item].values
            pred_W = round(sum(preds * weights), 3)
            dict_hybrid[item] = pred_W
        return dict_hybrid

    def get_top_N(self, topn):
        """Dictionary with topN items per algorithm."""
        df_all = self.get_all_predictions()
        d = {}
        for i in range(3):
            x = pd.DataFrame(df_all.iloc[i, 1:])
            d[df_all.iloc[i, 0]] = list(x.sort_values(
                by=self.user.id, axis=0, ascending=False).iloc[0:topn, 0].index)
        return d

    def compare(self, weights, topn, number_items_test):
        name_row = 'hit_{}'.format(topn)
        df_hit = pd.DataFrame(index=[name_row], columns=['CB', 'MF', 'CF'])
        cb_hits = []
        mf_hits = []
        cf_hits = []
        hy_hits = []
        if number_items_test == 'all':
            items_test = list(self.user.items_to_ignore)
        # elif number_items_test > len(self.user.items_to_ignore):
        #     print("NUMBER OF ITEMS TO TEST MUST BE SMALLER THAN {}".format(
        #         len(self.user.items_to_ignore)))
        else:
            items_test = list(self.user.items_to_ignore)[0:number_items_test]
        df_res = pd.DataFrame(index=items_test, columns=[
                              'ORIGINAL', 'CB', 'MF', 'CF', 'HY'])
        for i, IT in enumerate(items_test):
            print("{} of {} items".format(i, len(items_test)))
            new_utility = self.user.utility_matrix.copy()
            new_utility.loc[self.user.id, IT] = 0.0
            new_interactions = self.user.interactions_df.drop(self.user.interactions_df[(
                self.user.interactions_df.personId == self.user.id) & (self.user.interactions_df.contentId == IT)].index)
            new_user_object = User(self.user.id, new_utility, new_interactions)
            nhy = HYBRID(new_user_object, self.factors_mf,
                         self.N_cf, self.all_items, self.tfidf_matrix)
            ndf_all = nhy.get_all_predictions()
            dict_hybrid = nhy.get_hybrid_predictions(weights)
            ordered_keys = list({k: v for k, v in sorted(
                dict_hybrid.items(), key=lambda item: item[1], reverse=True)}.keys())
            top_hybrid = ordered_keys[0:topn]
            df_res.loc[IT, 'ORIGINAL'] = self.user.utility_matrix.loc[self.user.id, IT]/5.0
            df_res.loc[IT, 'CB'] = ndf_all[IT].values[0]
            df_res.loc[IT, 'MF'] = ndf_all[IT].values[1]
            df_res.loc[IT, 'CF'] = ndf_all[IT].values[2]
            df_res.loc[IT, 'HY'] = dict_hybrid[IT]
            dict_top = nhy.get_top_N(topn)
            if IT in dict_top['CB']:
                cb_hits.append(IT)
            if IT in dict_top['MF']:
                mf_hits.append(IT)
            if IT in dict_top['CF']:
                cf_hits.append(IT)
            if IT in top_hybrid:
                hy_hits.append(IT)

        df_hit.loc[name_row, 'CB'] = len(cb_hits)
        df_hit.loc[name_row, 'MF'] = len(mf_hits)
        df_hit.loc[name_row, 'CF'] = len(cf_hits)
        df_hit.loc[name_row, 'HY'] = len(hy_hits)
        return df_res, df_hit
