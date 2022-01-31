import numpy as np
import pandas as pd
from data_cleaning import getCSV, load, UserVec, ItemVec, store, prepareData, plot




class MatrixFact:
    def __init__(self, ratings_df, iterations, latent_vec_size=10, l4=0.02, gamma=0.005,
                 verbose=False, train_test_split=False):
        """
        :param ratings_df: padas dataframe containing the original user ratings
        :param iterations: number of iterations to carry on the SGD for
        :param l4: regurlatised squared errror constant
        :param gamma: SGD constant for size of incrimental update
        """
        self.verbose = verbose

        self.ratings_df = ratings_df
        if train_test_split:
            self.train_test_sets = load("data/temp/experiments/split_sets.DataSets.obj")
        else:
            self.train_test_sets = None

        self.latent_vec_size = latent_vec_size
        self.iterations = iterations

        self.l4 = l4
        self.gamma = gamma

        self.users_n, self.item_n = self.ratings_df["userId"].unique().shape[0], \
                                    self.ratings_df["movieId"].unique().shape[0]

        self.user_n_to_index = self.ratings_df["userId"].unique()
        self.movie_n_to_index = self.ratings_df["movieId"].unique()

        self.user_bias, self.item_bias = np.zeros(self.users_n), np.zeros(self.item_n)
        self.global_bias = np.mean(self.ratings_df.loc[self.ratings_df["rating"] != 0])["rating"]

        self.user_latent_v = np.random.normal(size=(self.users_n, self.latent_vec_size)) / self.latent_vec_size
        self.item_latent_v = np.random.normal(size=(self.item_n, self.latent_vec_size)) / self.latent_vec_size

    def save(self, path=None):
        # store(self.plots, "data/temp/plots.pkl")
        if path is None:
            store(self, "data/temp/MatrixFact_10.pkl")
        else:
            store(self, path)

    def originalRating(self, i, u):
        return \
            self.ratings_df.loc[(self.ratings_df["movieId"] == i) & (self.ratings_df["userId"] == u)]["rating"].values[
                0]

    def regularisedSquaredError(self, prediction, true_predictions, user_i, item_i):
        b_u = self.user_bias[user_i]
        b_i = self.item_bias[item_i]
        q_i = self.user_latent_v[user_i]
        p_u = self.item_latent_v[item_i]

        return (true_predictions - prediction) ** 2 + self.l4 * (
                b_u ** 2 + b_i ** 2 + np.linalg.norm(q_i) ** 2 + np.linalg.norm(p_u) ** 2)[0]

    def seenPredictions(self, ratings_df=None):
        if ratings_df is None:
            ratings_df = self.ratings_df

        def pred(row):
            user_i = np.where(self.user_n_to_index == row["userId"])[0]
            item_i = np.where(self.movie_n_to_index == row["movieId"])[0]
            prediction = self.predict(user_i, item_i)

            dct = {"userId": row["userId"],
                   "itemId": row["movieId"],
                   "rating": row["rating"],
                   "prediction": prediction,
                   "reg_sqr_err": self.regularisedSquaredError(prediction, row["rating"], user_i, item_i)}

            return pd.Series(dct)

        predictions = ratings_df.apply(lambda row: pred(row), axis=1)
        return predictions

    def allPredictions(self, user_id=None, alternate_df=None):
        """
        predics all preddicrions for seen and unseen movies for a specific user
        :param user_id:
        :return:
        """
        if alternate_df is None:
            ratings_df = self.ratings_df
        else:
            ratings_df = alternate_df

        movie_ids = pd.DataFrame(ratings_df["movieId"].unique())

        all_preds = []

        def pred(row):
            user_i = np.where(self.user_n_to_index == user_id)[0]
            item_i = np.where(self.movie_n_to_index == row[0])[0]
            prediction = self.predict(user_i, item_i)

            dct = {"userId": user_id,
                   "itemId": row[0],
                   "prediction": prediction}

            return pd.Series(dct)

        predictions = movie_ids.apply(lambda row: pred(row), axis=1)

        return predictions

    def SDE(self, alternate_df=None):
        if alternate_df is None:
            rating_df = self.ratings_df
        else:
            rating_df = alternate_df

        row_i = 1
        for _, row in rating_df.iterrows():
            if self.verbose:
                # print(f"{int(row_i * 100 / self.ratings_df.shape[0])}%")
                row_i += 1
            record = row
            userId, movieId, r = record["userId"], record["movieId"], record["rating"]
            user_i, item_i = np.where(self.user_n_to_index == userId)[0], np.where(self.movie_n_to_index == movieId)[0]

            prediction = self.predict(user_i, item_i)
            e = (r - prediction)

            # copy part of user vector as used in updating item vector
            user_bais_i_cpy = self.user_latent_v[user_i]

            # update all
            self.user_bias[user_i] += self.gamma * (e - self.l4 * self.user_bias[user_i])
            self.item_bias[item_i] += self.gamma * (e - self.l4 * self.item_bias[item_i])

            self.user_latent_v[user_i] += self.gamma * (
                    e * self.item_latent_v[item_i] - self.l4 * self.user_latent_v[user_i])
            self.item_latent_v[item_i] += self.gamma * (e * user_bais_i_cpy - self.l4 * self.item_latent_v[item_i])

    def user_info_prediction(self, user_i):
        user_i = int(user_i)
        mat_mul = self.user_latent_v[user_i].dot(self.user_additional_latent_v[user_i].T)
        prediction = mat_mul + np.mean(self.user_latent_v.dot(self.user_additional_latent_v.T))
        return prediction

    def learnModelParams(self):
        points = []
        train_test_flag = False
        if self.train_test_sets is not None:
            train_test_flag = True
            test, train = next(self.train_test_sets.genCrossFoldGroups())

        for train_iteration in range(self.iterations):
            print(f"{train_iteration}/{self.iterations}")

            if train_test_flag:
                self.SDE(alternate_df=train)
                # roughly 15 sencods per epock
                predicitons = self.seenPredictions(ratings_df=test)
            else:
                self.SDE()

            if train_test_flag:

                regularised_squared_error = sum(predicitons["reg_sqr_err"])
                avg_regularised_squared_error = sum(predicitons["reg_sqr_err"]) / predicitons.shape[0]
                print(f"{regularised_squared_error}, {avg_regularised_squared_error}")
                points.append((train_iteration, avg_regularised_squared_error))
                #plot(points)
        pass

    def predict(self, user_i, item_i):
        user_i = int(user_i)
        item_i = int(item_i)

        b_u = self.user_bias[user_i]
        b_i = self.item_bias[item_i]
        b_glob = self.global_bias
        mat_mul = self.user_latent_v[user_i].dot(self.item_latent_v[item_i].T)

        prediction = b_u + b_i + b_glob + mat_mul

        return prediction




def factoriseMatrix(load_matrix=False, save_path=None, ratings=None, iterations=32, train_test_split=False):
    # after gone though pre-procesesing
    if ratings is None:
        ml_ratings = getCSV("data/ml-latest-small/ratings.csv")
    else:
        ml_ratings = ratings

    if not load_matrix:
        mat = MatrixFact(ml_ratings, iterations=iterations, latent_vec_size=10, l4=0.02, gamma=0.005,
                         verbose=True, train_test_split=train_test_split)
        mat.learnModelParams()

    else:
        mat = load("data/temp/MatrixFact_30_OPT.pkl")

    if save_path is not None:
        mat.save(save_path)

    return mat


if __name__ == '__main__':
    #most genral number of iterations is 30
    factoriseMatrix(train_test_split=True, iterations=30)
