from sklearn.metrics.pairwise import linear_kernel
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import numpy as np
from scipy.stats.stats import pearsonr

from data_cleaning import prepareData, DataSets
from contentBased import ContentCompare
from data_cleaning import load, store, load
from MatrixFactorisation import factoriseMatrix, MatrixFact
from NeuralCollabertiveFiltering import neauralCollaberativeModel


class CreateExperiments:
    def __init__(self, whole_user_df, movie_df, load_sets=True):
        self.whole_user_df = whole_user_df
        self.movie_df = movie_df
        self.base_dir = "data/temp/experiments/"

        if not load_sets:
            self.train_test_sets = DataSets(self.movie_df, self.whole_user_df, 5)
            self.save(self.train_test_sets, self.base_dir + "split_sets.DataSets2.obj")
        else:
            self.train_test_sets = load(self.base_dir + "split_sets.DataSets.obj")

    def save(self, obj, path):
        # store(self.plots, "data/temp/plots.pkl")
        store(obj, path)

    def load(self, path):
        return load("data/temp/genres.pkl")

    def contentBasedTrainModels(self):
        content_based = ContentCompare(self.movie_df, self)

    def matrixFactorTrainModels(self):
        group_num = 0
        for train, test in self.train_test_sets.genCrossFoldGroups():
            mat = factoriseMatrix(load_matrix=False, ratings=train)
            self.save(mat, self.base_dir + f"group_{group_num}.matrixFact.obj")

    def NCFPTrainModels(self):
        NCM = neauralCollaberativeModel(load_mat=True, load_model=False, ratings=self.whole_user_df.ratings)


if __name__ == '__main__':
    items, users = prepareData(load_stored_data=True, reduce=True, min_user_reviews=100, min_movie_raings=50)
    experiments = CreateExperiments(users, items, load_sets=True)
    experiments.matrixFactorTrainModels()
