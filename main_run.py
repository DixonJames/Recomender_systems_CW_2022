from sklearn.metrics.pairwise import linear_kernel
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import numpy as np
from scipy.stats.stats import pearsonr

from data_cleaning import prepareData
from contentBased import ContentCompare
from data_cleaning import load
from MatrixFactorisation import factoriseMatrix, MatrixFact
from NeuralCollabertiveFiltering import neauralCollaberativeModel


class User:
    def __init__(self, whole_user_df, movie_df, user_id):
        self.whole_user_df = whole_user_df
        self.movie_df = movie_df
        self.user_id = user_id

        self.user_df = whole_user_df.user_df[whole_user_df.user_df["userId"] == self.user_id]
        self.user_reviews = whole_user_df.ratings[whole_user_df.ratings["userId"] == self.user_id]

        self.content_based = ContentCompare(movie_df, self)

        self.user_mean_score_diff = -whole_user_df.user_df["mean_score"].mean() + float(
            whole_user_df.user_df[whole_user_df.user_df["userId"] == self.user_id]["mean_score"])

        self.movie_df = movie_df.clean_items

    def ratingWeighting(self, rating):
        """
        so good it rhymes
        ratings one below avarage is a bad rating
        weight_dict = {-4:-1, -3:-0.6, -2:-0.3, -1:0, 0:0.3, 1:0.6, 2:1}
        """
        weight_dict = {-5: -1, -4: -1, -3: -1, -2: -0.6, -1: -0.3, 0: 0.3, 1: 0.6, 2: 1, 3: 1, 4: 1, 5: 1}
        user_mean_score = int(float(self.user_df["mean_score"]) * 5)
        score_diff = int(user_mean_score - rating)
        return weight_dict[score_diff]

    def createUserVector(self):
        """
        goes though all users movie reviews.
        creates average movie vecotor
        weighted by users rating
        """
        user_vec = np.zeros_like(np.array(self.movie_df.iloc[int(1)]))
        for index, row in self.user_reviews.iterrows():
            rating = row["rating"]
            movie_id = row["movieId"]
            movie_vec = np.array(self.movie_df.iloc[int(movie_id)])
            weighting = self.ratingWeighting(rating)

            user_vec += (weighting * movie_vec)
        user_vec = user_vec / self.user_reviews.shape[0]

        return user_vec

    def createClassUserVecotor(self):
        user_vec_classes = [np.zeros_like(np.array(self.movie_df.iloc[int(1)])) for _ in range(5)]
        user_weight_counts = [(self.user_reviews[self.user_reviews["rating"] == i + 1].shape[0]) for i in range(5)]
        user_weight_classes = [1 / (1 + sorted(user_weight_counts, reverse=True).index(i)) for i in user_weight_counts]

        for index, row in self.user_reviews.iterrows():
            rating = int(row["rating"])
            movie_id = row["movieId"]
            movie_vec = np.array(self.movie_df.iloc[int(movie_id)])

            user_vec_classes[rating - 1] += movie_vec

        for user_vec_i in range(len(user_vec_classes)):
            user_vec_classes[user_vec_i] = user_vec_classes[user_vec_i] / user_vec_classes[user_vec_i].shape[0]

        return user_vec_classes, user_weight_classes

    def contentBasedPrediction(self):
        all_predicitons = self.content_based.queryUser(self.user_id, profile_vector_type="classed",
                                                       distance_measure="cosine")
        return all_predicitons

    def matrixFactorPrediction(self):
        mat = factoriseMatrix(load=False, ratings=self.whole_user_df.ratings)
        all_predictions = mat.allPredictions(self.user_id)
        return all_predictions

    def NCFPPrediction(self):
        NCM = neauralCollaberativeModel(load_mat=True, load_model=True, ratings=self.whole_user_df.ratings)
        all_predictions = NCM.allPredictions(self.user_id)
        return all_predictions


if __name__ == '__main__':
    items, users = prepareData(load_stored_data=True, reduce=True, min_user_reviews=100, min_movie_raings=50)
    test_u = User(whole_user_df=users, movie_df=items, user_id=1)
    test_u.NCFPPrediction()
