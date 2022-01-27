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

        if MF_model is not None:
            self.MF_model = MF_model
        else:
            self.MF_model = factoriseMatrix(load_matrix=False, ratings=self.whole_user_df.ratings)

        if NCF_model is not None:
            self.NCF_model = NCF_model
        else:
            self.NCF_model = neauralCollaberativeModel(load_model=False, ratings=self.whole_user_df.ratings)

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
        content_based = ContentCompare(self.whole_movie_df, self)
        all_predictions = content_based.queryUser(profile_vector_type="weighted", distance_measure="cosine")
        return all_predictions

    def matrixFactorPrediction(self):
        all_predictions = self.MF_model.allPredictions(self.user_id)
        return all_predictions

    def NCFPPrediction(self):
        all_predictions = self.NCF_model.allPredictions(self.user_id)
        return all_predictions


class RecommenderSystem:
    def __init__(self):
        self.items, self.users = prepareData(load_stored_data=True, reduce=True, min_user_reviews=100,
                                             min_movie_raings=50)

        self.current_user = None

    def startUp(self):
        """
        first thign to run
        inincal user login
        runs main menue for first time
        :return:
        """
        print("======== Welcome ==========")
        self.selectUser()
        self.mainMenue()

    def mainMenue(self):
        print("===========================")
        print("========== Menu ===========")
        print("===========================")

        print("1)   select user login")
        print("2)   get recommendations")
        print("3)   crete new rating")
        print("4)   quit session")

        option = self.IOResponse("Enter selection", [1, 2, 3, 4])
        if option == 1:
            self.selectUser()
        elif option == 2:
            self.generateRecomendations()
        elif option == 3:
            self.createRating()
        elif option == 4:
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
        return response

    def createRating(self):
        movieId = "N"
        i = 0
        while movieId != "N":
            disp_movies = self.items.genres["movieId"].unique()[:5 + i]
            print(disp_movies)
            print("enter 'N' for next set of movies ")
            movieId = self.IOResponse("please select existing user ID", disp_movies.append("N"))
            i += 1

        # create rating
        rating = self.IOResponse("please enter a movie rating", [1, 2, 3, 4, 5])
        timestamp = datetime.datetime(2022, 30, 1).timestamp()

        self.users.ratings.append({"userId": self.current_user.user_id})

    def warmUpRatings(self):
        """
        gets a new user to create some warm up ratings
        :return:
        """
        print("===== Rating 5 movies =====")
        for i in range(5):
            self.createRating()

    def createNewUser(self):
        """
        adds new user to dataframes
        returns new users ID
        """
        new_id = max(self.users.user_df["userId"].unique()) + 1

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
                disp_users = list(self.users.user_df["userId"].unique()[5*(i-1):5 * i])
                print("enter 'next' for next set of user IDs")
                disp_users.append("next")
                userId = self.IOResponse("please select existing user ID", disp_users)
                if userId == "next":
                    i += 1
                else:
                    selected = True
        userId = int(userId)

        self.current_user = User(whole_user_df=self.users, movie_df=self.items, user_id=userId)

    def generateRecomendations(self):
        """
        select desired recommender system
        churns out as many responses as required
        :return:
        """
        pass


if __name__ == '__main__':
    # items, users = prepareData(load_stored_data=True, reduce=True, min_user_reviews=100, min_movie_raings=50)
    # test_u = User(whole_user_df=users, movie_df=items, user_id=1)
    rs = RecommenderSystem()
    rs.startUp()
