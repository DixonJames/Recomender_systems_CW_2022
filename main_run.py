import random
from tabulate import tabulate
import sys

from sklearn.metrics.pairwise import linear_kernel
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import numpy as np
from datetime import datetime
from scipy.stats.stats import pearsonr

from data_cleaning import prepareData
from contentBased import ContentCompare
from data_cleaning import load
from MatrixFactorisation import factoriseMatrix, MatrixFact
from NeuralCollabertiveFiltering import neauralCollaberativeModel
from hybrid_rs import Mf_Collaberative_hybrid

MF_model_path = "data/temp/main_use/group_4.matrixFact.obj"
NCM_model_path = "data/temp/main_use/group_4.NCM.obj"

today_date = datetime(2022, 1, 30)


class User:
    def __init__(self, whole_user_df, movie_df, user_id, MF_model=None, NCF_model=None):
        self.whole_user_df = whole_user_df
        self.whole_movie_df = movie_df
        self.user_id = user_id

        self.user_df = whole_user_df.user_df[whole_user_df.user_df["userId"] == self.user_id]
        self.user_reviews = whole_user_df.ratings[whole_user_df.ratings["userId"] == self.user_id]

        self.user_mean_score_diff = -whole_user_df.user_df["mean_score"].mean() + float(
            whole_user_df.user_df[whole_user_df.user_df["userId"] == self.user_id]["mean_score"])

        self.movie_df = movie_df.clean_items

        self.MF_model = MF_model
        self.NCF_model = NCF_model


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
            movie_vec = np.array(self.movie_df.loc[int(movie_id)])
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
            try:
                movie_vec = np.array(self.movie_df.loc[int(movie_id)])
            except:
                pass

            user_vec_classes[rating - 1] += movie_vec

        for user_vec_i in range(len(user_vec_classes)):
            user_vec_classes[user_vec_i] = user_vec_classes[user_vec_i] / user_vec_classes[user_vec_i].shape[0]

        return user_vec_classes, user_weight_classes

    def contentBasedPrediction(self):
        content_based = ContentCompare(user_v=self.createUserVector(), class_user_v=self.createClassUserVecotor())
        all_predictions = content_based.queryUser(profile_vector_type="weighted", distance_measure="cosine")
        return all_predictions

    def matrixFactorPrediction(self):
        if self.MF_model is None:
            self.MF_model = factoriseMatrix(load_matrix=False, iterations=30)

        all_predictions = self.MF_model.allPredictions(self.user_id)
        return all_predictions

    def hybridPrediction(self, CB_weighting=0.5):

        hybrid_model = Mf_Collaberative_hybrid(CB_pred=self.contentBasedPrediction(), MF_pred=self.matrixFactorPrediction(), reviews=self.user_reviews, CB_weighting=CB_weighting)
        all_predictions = hybrid_model.allPredictions()

        return all_predictions

    def NCFPPrediction(self):
        if self.NCF_model is None:
            self.NCF_model = neauralCollaberativeModel(load_model=False)

        all_predictions = self.NCF_model.allPredictions(self.user_id)
        return all_predictions

    def hybrid_recommendations(self, number_recs=30, CB_weighting=0.5):

        hybrid_model = Mf_Collaberative_hybrid(CB_pred=self.contentBasedPrediction(),
                                               MF_pred=self.matrixFactorPrediction(), reviews=self.user_reviews,
                                               CB_weighting=CB_weighting)
        all_recommendations = (hybrid_model.rankCandidates(number_recs=number_recs))["itemId"].values

        return all_recommendations

    def NCF_recommendations(self, number_recs=30):
        if self.NCF_model is None:
            self.NCF_model = neauralCollaberativeModel(load_model=False)

        all_recommendations = self.NCF_model.allPredictions(self.user_id)
        all_recommendations.sort_values(by="prediction_NCF", ascending=False)

        return (all_recommendations.head(number_recs))["itemId"].values.astype(int)


class RecommenderSystem:
    def __init__(self, MF_model_path, NCM_model_path):
        self.items, self.users = prepareData(load_stored_data=True, reduce=True, min_user_reviews=100,
                                             min_movie_raings=50)
        self.MF_model_path = MF_model_path
        self.NCM_model_path = NCM_model_path

        self.current_user = None
        self.current_user_id = None

    def startUp(self):
        """
        first thign to run
        inincal user login
        runs main menue for first time
        :return:
        """
        print("======== Welcome ==========")
        self.selectUser()

    def mainMenue(self):
        print("===========================")
        print("========== Menu ===========")
        print("===========================")

        print("1)   select user login")
        print("2)   get recommendations")
        print("3)   crete new rating")
        print("4)   quit session")

        option = self.IOResponse("Enter selection", ["1", "2", "3", "4"])
        if option == "1":
            self.selectUser()
        elif option == "2":
            self.generateRecomendations()
        elif option == "3":
            self.createRating()
        elif option == "4":
            sys.exit()

    def IOResponse(self, message, options=None):

        response = "_"
        if options is not None:
            options = [str(i) for i in options]
            print(f"Valid inputs: {options}")
            while response not in options:
                response = input(f"{message}:")
        else:
            response = input(f"{message}:")

        print()
        return response.upper()

    def createRating(self, return_menue=True):
        movieId = "N"
        i = 1
        while movieId == "N":
            disp_movies = self.items.genres["movieId"].unique()[5 * (i - 1):5 * i]
            output_df = self.items.genres[self.items.genres["movieId"].isin(disp_movies)][["movieId", "title"]]
            print(tabulate(output_df, headers='keys', tablefmt='psql', showindex=False))

            print("enter 'N' for next set of movies ")
            disp_movies = list(disp_movies)
            disp_movies.extend(["N", "n"])
            movieId = self.IOResponse("please enter a Movie ID", disp_movies)
            i += 1

        # create rating
        rating = self.IOResponse("please enter a movie rating", [1, 2, 3, 4, 5])
        timestamp = today_date.timestamp()

        self.users.ratings = self.users.ratings.append(
            {"userId": self.current_user_id, "movieId": movieId, "rating": rating, "timestamp": timestamp},
            ignore_index=True
        )

        if return_menue:
            self.mainMenue()

    def warmUpRatings(self):
        """
        gets a new user to create some warm up ratings
        :return:
        """
        print("===== Rating 5 movies =====")
        for i in range(5):
            self.createRating(return_menue=False)

        self.mainMenue()

    def createNewUser(self):
        """
        adds new user to dataframes
        returns new users ID
        """
        new_id = max(self.users.user_df["userId"].unique()) + 1
        self.current_user_id = new_id
        self.warmUpRatings()
        mean_score = self.users.ratings[self.users.ratings["userId"] == self.current_user_id]["rating"].mean()
        review_num = self.users.ratings[self.users.ratings["userId"] == self.current_user_id].shape[0] / max(
            self.users.ratings["userId"].value_counts())

        reviewed_movies = self.users.ratings[self.users.ratings["userId"] == 1]["movieId"].values
        for movieId in reviewed_movies:
            self.users.reviews.loc[self.users.reviews.index == movieId, 0] += 1

        self.users.user_df.append(
            {"userId": new_id, "mean_score": mean_score, "review_num": review_num, "age": random.uniform(0.1, 1.0),
             "country": random.uniform(0.1, 1.0)})

    def selectUser(self):
        """
        selects the current user.
        select pre_existing or create new one
        """
        print("===== SELECTING USER =====")
        NEW_USER_OPTION = self.IOResponse("Select: new(N) or existing(E) user", ["N", "n", "E", "e"])

        if NEW_USER_OPTION.upper() == "N":
            userId = self.createNewUser()

        else:
            # display potencail users
            selected = False
            i = 1
            while not selected:
                disp_users = list(self.users.user_df["userId"].unique()[5 * (i - 1):5 * i])
                print("enter 'next' for next set of user IDs")
                disp_users.append("next")
                userId = self.IOResponse("please select existing user ID", disp_users)

                if userId == "next":
                    i += 1
                else:
                    selected = True
        userId = int(userId)

        self.current_user_id = userId
        self.current_user = User(whole_user_df=self.users, movie_df=self.items, user_id=userId,
                                 MF_model=self.MF_model_path, NCF_model=self.NCM_model_path)

        self.mainMenue()

    def generateRecomendations(self):
        """
        select desired recommender system
        churns out as many responses as required
        :return:
        """

        print()
        print("===== CREATING RECOMMENDATIONS =====")
        df_selector = pd.DataFrame(columns=["Number", "System"])
        df_selector = df_selector.append({"Number": 1, "System": "Nural Collaborative Filtering"}, ignore_index=True)
        df_selector = df_selector.append({"Number": 2, "System": "Matrix factorisation & Content based hybrid"},
                                         ignore_index=True)
        print(tabulate(df_selector, headers='keys', tablefmt='psql', showindex=False))

        self.IOResponse("Select desired recommender system", ["1", "2"])

        self.mainMenue()


if __name__ == '__main__':
    # items, users = prepareData(load_stored_data=True, reduce=True, min_user_reviews=100, min_movie_raings=50)
    # test_u = User(whole_user_df=users, movie_df=items, user_id=1)
    rs = RecommenderSystem(MF_model_path=MF_model_path, NCM_model_path=NCM_model_path)
    # rs.createNewUser()
    rs.startUp()
