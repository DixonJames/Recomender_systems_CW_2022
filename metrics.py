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


class Serendipity:
    def __init__(self, users, items):
        """

        :param users:
        :param items:
        :param recomendations: for each user a set of ratings for each movie
        """
        self.users = users
        self.items = items

    def p_single(self, i):
        """
        prob of ANY user interacting with item of index i
        unmberof usersin interacting with i / number of users
        :return: probability item interacted with
        """
        interacted = (self.users.ratings[self.users.ratings["movieId"] == i])["userId"].unique().shape[0]
        user_num = self.users.ratings["userId"].unique().shape[0]
        return interacted / user_num

    def p_double(self, i, j):
        """
        prob of ANY user interacting with items of index i and j
        unmber of users in interacting with i and j / number of users
        :return: probability item interacted with
        """
        interacted_i = (self.users.ratings[self.users.ratings["movieId"] == i])["userId"].unique()
        interacted_j = (self.users.ratings[self.users.ratings["movieId"] == j])["userId"].unique()

        interacted_i_j = list(set(list(interacted_j)).intersection(set(list(interacted_i))))

        user_num = self.users.ratings["userId"].unique().shape[0]

        return len(interacted_i_j) / user_num

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
        vec_i = self.items.clean_items.loc[index_i].values
        vec_j = self.items.clean_items.loc[index_j].values

        return cosine_similarity([vec_i, vec_j]).min()

    def itemSurprise(self, i, H):
        """
        max PMI of i with each h in uesrs history H
        :param i:
        :param H:
        :return:
        """
        return max([self.PMI(i, h) for h in H])

    def unexpectedness(self, I, H, metric="cos"):
        """
        measuemtn of 'surprise' when recomending set I compared to history H
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

            h_vals = h_vals / len(H)
            i_vals += h_vals
        i_vals = i_vals / len(I)

        return i_vals

    def relevance(self, interactions, item):
        if item in interactions:
            return 1
        return 0

    def userSerendipity(self, historic_interactions, recomendations):
        """
        for one user
        :param test_set:
        :return:
        """
        def serendipitySingle(i):
            return self.relevance(interactions=historic_interactions, item=i) * self.unexpectedness([i],
                                                                                                    historic_interactions,
                                                                                                    metric="PMI")

        serendipitySingle_v = np.vectorize(serendipitySingle)
        #tot = sum(serendipitySingle_v(recomendations))
        tot = 0
        for i in recomendations:
            if i in set(recomendations):
                relevance = 1
            else:
                relevance = 0
            tot += relevance * self.unexpectedness([i], historic_interactions, metric="PMI")

        #tot = sum([self.relevance(interactions=historic_interactions, item=i) * self.unexpectedness([i],historic_interactions,metric="PMI") for i in recomendations])

        return tot / len(recomendations)


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

        for test, train in self.train_test_sets.genCrossFoldGroups():
            mat = factoriseMatrix(load_matrix=False, ratings=train, iterations=30)
            store(mat, self.base_dir + f"group_{group_num}.matrixFact.obj")

            NCM = neauralCollaberativeModel(load_mat=False, pass_mat=mat, load_model=False, epoch_num=4,
                                            train_test_split=False, ratings=train)
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

        self.serendipity = Serendipity(users=self.whole_user_df, items=self.whole_movie_df)

    def reducedReviewDf(self, userId, test_userReviews):
        """
        reaplces one users reviews with the ones only in their test set
        :param userId:
        :param test_userReviews:
        :return:
        """
        new_whole_user_df_ratings = pd.concat([test_userReviews[test_userReviews["userId"] == userId],
                                               self.whole_user_df.ratings[
                                                   self.whole_user_df.ratings["userId"] != userId]])
        return new_whole_user_df_ratings

    def run(self):
        overall_results = pd.DataFrame(
            columns=["userId", "user interactions", "absError_hybrid", "absError_NCF", "serendipity_CBF&MF_25",
                     "serendipity_CBF&MF_10", "serendipity_NCF_25", "serendipity_NCF_10"])
        user_i = 1
        user_num = len(self.whole_user_df.ratings["userId"].unique())
        for userIndex in self.whole_user_df.ratings["userId"].unique():
            historic_interactions = (self.whole_user_df.ratings[self.whole_user_df.ratings["userId"] == userIndex])[
                "movieId"].values
            print(f"{100 * (user_i / user_num)}%")
            user_i += 1

            individual_results = pd.DataFrame(columns=["movieId", "CBF&MF", "NCF"])
            group_i = 0

            mat_model = load(f"data/temp/experiments/group_{group_i}.matrixFact.obj")
            ncf_model = load(f"data/temp/experiments/group_{group_i}.NCM.obj")

            overall_user = User(whole_user_df=self.whole_user_df, movie_df=self.whole_movie_df, user_id=userIndex,
                                MF_model=mat_model, NCF_model=ncf_model)

            serendipity = {"serendipity_CBF&MF_25": 0, "serendipity_CBF&MF_10": 0, "serendipity_NCF_25": 0,
                           "serendipity_NCF_10": 0}


            for test, train in self.train_test_sets.genCrossFoldGroups():


                mat_model = load(f"data/temp/experiments/group_{group_i}.matrixFact.obj")
                ncf_model = load(f"data/temp/experiments/group_{group_i}.NCM.obj")

                test_user = User(whole_user_df=self.whole_user_df, movie_df=self.whole_movie_df, user_id=userIndex,
                                 MF_model=mat_model, NCF_model=ncf_model)
                test_user.user_reviews = self.reducedReviewDf(userIndex, train)

                # results_CBF = test_user.contentBasedPrediction()
                # results_MF = test_user.matrixFactorPrediction()

                results_HYBRID = test_user.hybridPrediction()
                results_NCF = test_user.NCFPPrediction()

                recs_HYBRID_25 = test_user.hybrid_recommendations(number_recs=25)
                recs_NCF_25 = test_user.NCF_recommendations(number_recs=25)
                recs_HYBRID_10 = test_user.hybrid_recommendations(number_recs=10)
                recs_NCF_10 = test_user.NCF_recommendations(number_recs=10)

                all_results = pd.merge(results_NCF, results_HYBRID, on='itemId', how='outer')
                test_results = all_results[
                    all_results["itemId"].isin(list(test[test["userId"] == userIndex]["movieId"]))]

                # now get recomendation, minus the results in the current train set

                formatted_results = pd.DataFrame()
                for col_name, col_data in zip(["movieId", "CBF&MF", "NCF"],
                                              [test_results["itemId"].values, test_results["prediction_hybrid"].values,
                                               test_results["prediction_NCF"].values]):
                    formatted_results[col_name] = col_data

                individual_results = pd.concat([individual_results, formatted_results])

                serendipity["serendipity_CBF&MF_25"] += self.serendipity.userSerendipity(
                    historic_interactions=historic_interactions, recomendations=recs_HYBRID_25)
                serendipity["serendipity_CBF&MF_10"] += self.serendipity.userSerendipity(
                    historic_interactions=historic_interactions, recomendations=recs_HYBRID_10)
                serendipity["serendipity_NCF_25"] += self.serendipity.userSerendipity(
                    historic_interactions=historic_interactions, recomendations=recs_NCF_25)
                serendipity["serendipity_NCF_10"] += self.serendipity.userSerendipity(
                    historic_interactions=historic_interactions, recomendations=recs_NCF_10)

                group_i += 1

            individual_results = pd.merge(self.whole_user_df.ratings[self.whole_user_df.ratings["userId"] == userIndex],
                                          individual_results,
                                          on='movieId', how='outer')
            individual_results["absError_hybrid"] = abs(individual_results["rating"] - individual_results["CBF&MF"])
            individual_results["absError_NCF"] = abs(individual_results["rating"] - individual_results["NCF"])

            users_ratings = (self.whole_user_df.ratings[self.whole_user_df.ratings["userId"] == userIndex])[
                "movieId"].values

            serendipity["serendipity_CBF&MF_25"] = serendipity["serendipity_CBF&MF_25"] / group_i

            serendipity["serendipity_CBF&MF_10"] = serendipity["serendipity_CBF&MF_10"] / group_i

            serendipity["serendipity_NCF_25"] = serendipity["serendipity_NCF_25"] / group_i

            serendipity["serendipity_NCF_10"] = serendipity["serendipity_NCF_10"] / group_i

            overall_results.append({"userId": userIndex,
                                    "user interactions": len(users_ratings),
                                    "absError_hybrid": individual_results["absError_hybrid"],
                                    "absError_NCF": individual_results["absError_NCF"],
                                    "serendipity_CBF&MF_25": serendipity["serendipity_CBF&MF_25"],
                                    "serendipity_CBF&MF_10": serendipity["serendipity_CBF&MF_10"],
                                    "serendipity_NCF_25": serendipity["serendipity_NCF_25"],
                                    "serendipity_NCF_10": serendipity["serendipity_NCF_10"]},
                                   ignore_index=True)

            store(overall_results, "data/temp/experiments/overall_results.obj")
        return overall_results

    def display(self):
        overall_results = load("data/temp/experiments/overall_results.obj")



def setupModels():
    items, users = prepareData(load_stored_data=True, reduce=True, min_user_reviews=100, min_movie_raings=50)
    experiments = CreateExperiments(users, items, load_sets=True)
    experiments.TrainAllPartialModels()


if __name__ == '__main__':
    # setupModels()
    items, users = prepareData(load_stored_data=True, reduce=True, min_user_reviews=100, min_movie_raings=50)

    experiments = runExperiemts(users, items, load_sets=True)
    #experiments.run()
    experiments.display()
