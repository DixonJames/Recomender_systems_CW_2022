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
from main_run import User
from math import log2


class Unexpectedness:
    def __init__(self, users, items, recomendations):
        """

        :param users:
        :param items:
        :param recomendations: for each user a set of ratings for each movie
        """
        self.users = users
        self.items = items
        self.recomendations = recomendations

    def p_single(self, i):
        """
        prob of ANY user interacting with item of index i
        unmberof usersin interacting with i / number of users
        :return: probability item interacted with
        """
        interacted = (self.users.reviews[self.users.reviews["movieId"] == i])["userId"].unique().shape[0]
        user_num = self.users.reviews["userId"].unique().shape[0]
        return interacted/user_num

    def p_double(self, i, j):
        """
        prob of ANY user interacting with items of index i and j
        unmber of users in interacting with i and j / number of users
        :return: probability item interacted with
        """
        interacted_i = (self.users.reviews[self.users.reviews["movieId"] == i])["userId"].unique()
        interacted_j = (self.users.reviews[self.users.reviews["movieId"] == j])["userId"].unique()

        interacted_i_j = list(set(list(interacted_j.values)).intersection(set(list(interacted_i.values))))

        user_num = self.users.reviews["userId"].unique().shape[0]

        return len(interacted_i_j)/user_num



    def PMI(self, i, j):
        """
        point-wise mutual information
        item similarity based on number of users that have interacted with both
        PMI(i,j)=−log2(p(i,j)/p(i)×p(j))/log2p(i,j)
        :param i: item index
        :param j: item index
        :return: -1 to 1,  -1 means items are never used together and vice versa
        """

        return -1 * (log2((self.p_double(i, j) / (self.p_single(i) * self.p_single(j)))) / log2(self.p_double(i, j)))

    def cosine_sim(self, index_i, index_j):
        # get item vecotrs for i and j
        vec_i = self.items.clean_items.iloc[index_i].values[0]
        vec_j = self.items.clean_items.iloc[index_j].values[0]

        return cosine_similarity(vec_i, vec_j)

    def unexpectedness(self, I, H, metric="cos"):
        """
        for a particular user
        :param I: users recommendations
        :param H: users historical interactions
        :param metric: cos/pmi
        :return:
        """
        i_vals = 0
        for i in I:
            h_vals = 0
            for h in H:
                if metric == "cos":
                    h_vals += self.cosine_sim(i, h)
                else:
                    h_vals += self.PMI(i, h)
            h_vals = h_vals/len(H)
            i_vals += h_vals
        i_vals = i_vals/len(i)

        return i_vals

    def scoreAll(self):
        for userId in self.users.reviews["userId"].unique:
            pass



class CreateExperiments:
    def __init__(self, whole_user_df, movie_df, load_sets=True):
        self.whole_user_df = whole_user_df
        self.movie_df = movie_df
        self.base_dir = "data/temp/experiments/"

        if not load_sets:
            self.train_test_sets = DataSets(self.movie_df, self.whole_user_df, 5)
            store(self.train_test_sets, self.base_dir + "split_sets.DataSets2.obj")
        else:
            self.train_test_sets = load(self.base_dir + "split_sets.DataSets.obj")

    def contentBasedTrainModels(self):
        content_based = ContentCompare(self.movie_df, self)

    def TrainAllPartialModels(self):
        group_num = 0
        for train, test in self.train_test_sets.genCrossFoldGroups():
            mat = factoriseMatrix(load_matrix=False, ratings=train, iterations=30)
            store(mat, self.base_dir + f"group_{group_num}.matrixFact.obj")

            NCM = neauralCollaberativeModel(load_mat=False, pass_mat=mat, load_model=False,  epoch_num=3)
            store(NCM, self.base_dir + f"group_{group_num}.NCM.obj")
            group_num += 1


class runExperiemts():
    def __init__(self, whole_user_df, movie_df, load_sets=True):
        self.whole_user_df = whole_user_df
        self.whole_movie_df = movie_df
        self.base_dir = "data/temp/experiments/"

        if not load_sets:
            self.train_test_sets = DataSets(self.whole_movie_df, self.whole_user_df, 5)
            store(self.train_test_sets, self.base_dir + "split_sets.DataSets2.obj")
        else:
            self.train_test_sets = load(self.base_dir + "split_sets.DataSets.obj")

    def createNewuserMovieDFs(self, userId, userReviews):
        new_whole_user_df_ratings = pd.concat(
            [self.whole_user_df.ratings[self.whole_user_df.ratings["userId"] != userId], userReviews])
        return new_whole_user_df_ratings

    def run(self):
        overall_results = pd.DataFrame(columns=["userId", "CBF", "MF", "NCF"])
        for userIndex in self.whole_user_df.ratings["userId"].unique():
            results = pd.DataFrame()
            group_i = 0
            for train, test in self.train_test_sets.genCrossFoldGroups():
                reduced_review_df = self.createNewuserMovieDFs(userIndex, test)

                mat_model = load(f"data/temp/experiments/group_{group_i}.matrixFact.obj")
                ncf_model = load(f"data/temp/experiments/group_{group_i}.NCM.obj")

                test_user = User(whole_user_df=self.whole_user_df, movie_df=self.whole_movie_df, user_id=userIndex,
                                 MF_model=mat_model, NCF_model=ncf_model)
                test_user.user_reviews = test[test["userId"] == userIndex]

                results_CBF = test_user.contentBasedPrediction()
                results_MF = test_user.matrixFactorPrediction()
                results_NCF = test_user.NCFPPrediction()

                group_i += 1

            overall_results.append({"userId": 0, "CBF": 0, "MF": 0, "NCF": 0})


def setupModels():
    items, users = prepareData(load_stored_data=True, reduce=True, min_user_reviews=100, min_movie_raings=50)
    experiments = CreateExperiments(users, items, load_sets=True)
    experiments.TrainAllPartialModels()


if __name__ == '__main__':
    setupModels()
    items, users = prepareData(load_stored_data=True, reduce=True, min_user_reviews=100, min_movie_raings=50)

    experiments = runExperiemts(users, items, load_sets=True)

    # experiments.run()
