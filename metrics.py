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
            mat = factoriseMatrix(load_matrix=False, ratings=train, iterations=1000)
            store(mat, self.base_dir + f"group_{group_num}.matrixFact.obj")

            NCM = neauralCollaberativeModel(load_mat=False, pass_mat=mat, load_model=False, ratings=train)
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
        new_whole_user_df_ratings = pd.concat([self.whole_user_df.ratings[self.whole_user_df.ratings["userId"]!=userId], userReviews])
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

                test_user = User(whole_user_df=self.whole_user_df, movie_df=self.whole_movie_df, user_id=userIndex, MF_model=mat_model, NCF_model=ncf_model)
                test_user.user_reviews = test[test["userId"] == userIndex]

                results_CBF = test_user.contentBasedPrediction()
                results_MF = test_user.matrixFactorPrediction()
                results_NCF = test_user.NCFPPrediction()

                group_i += 1

            overall_results.append({"userId":0, "CBF":0, "MF":0, "NCF":0})




def setupModels():
    items, users = prepareData(load_stored_data=True, reduce=True, min_user_reviews=100, min_movie_raings=50)
    experiments = CreateExperiments(users, items, load_sets=True)
    experiments.TrainAllPartialModels()

if __name__ == '__main__':
    setupModels()
    items, users = prepareData(load_stored_data=True, reduce=True, min_user_reviews=100, min_movie_raings=50)

    experiments = runExperiemts(users, items, load_sets=True)

    #experiments.run()

