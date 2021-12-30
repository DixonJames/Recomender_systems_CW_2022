import numpy as np
import pandas as pd
from data_cleaning import getCSV


class MatrixFact:
    def __init__(self, ratings_df, iterations, latent_vec_size=10, test_proportion=0.2, l4=0.02, gamma=0.005, verbose=False):
        """
        :param ratings_df: padas dataframe containing the original user ratings
        :param iterations: number of iterations to carry on the SGD for
        :param l4: regurlatised squared errror constant
        :param gamma: SGD constant for size of incrimental update
        """
        self.verbose = verbose
        self.ratings_df = ratings_df
        self.latent_vec_size = latent_vec_size
        self.iterations = iterations
        self.test_prop = test_proportion
        self.l4 = l4
        self.gamma = gamma

        self.users_n, self.item_n = self.ratings_df["userId"].unique().shape[0], \
                                    self.ratings_df["movieId"].unique().shape[0]

        self.user_n_to_index = self.ratings_df["userId"].unique()
        self.movie_n_to_index = self.ratings_df["movieId"].unique()

        self.user_bias, self.item_bias = np.zeros(self.users_n), np.zeros(self.item_n)

        self.user_v = np.random.normal(size=(self.users_n, self.latent_vec_size)) / self.latent_vec_size
        self.item_v = np.random.normal(size=(self.item_n, self.latent_vec_size)) / self.latent_vec_size

        self.global_bias = np.mean(self.ratings_df.loc[self.ratings_df["rating"] != 0])["rating"]

        self.learnModelParams()

    def originalRating(self, i, u):
        return \
            self.ratings_df.loc[(self.ratings_df["movieId"] == i) & (self.ratings_df["userId"] == u)]["rating"].values[
                0]

    def predict(self, user_i, item_i):
        user_i = int(user_i)
        item_i = int(item_i)

        b_u = self.user_bias[user_i]
        b_i = self.item_bias[item_i]
        b_glob = self.global_bias
        mat_mul = self.user_v[user_i].dot(self.item_v[item_i].T)

        prediction = b_u + b_i + b_glob + mat_mul

        return prediction

    def regularisedSquaredError(self, prediction, true_predictions, user_i, item_i):
        b_u = self.user_bias[user_i]
        b_i = self.item_bias[item_i]
        q_i = self.user_v[user_i]
        p_u = self.item_v[item_i]

        return (true_predictions - prediction) ** 2 + self.l4 * (
                    b_u ** 2 + b_i ** 2 + np.linalg.norm(q_i) ** 2 + np.linalg.norm(p_u) ** 2)[0]

    def allPredictions(self):
        all = np.zeros((self.users_n, self.item_n))

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

        predictions = self.ratings_df.apply(lambda row: pred(row), axis=1)
        return predictions

    def SDE(self):
        row_i = 1
        for _, row in self.ratings_df.iterrows():
            if self.verbose:
                #print(f"{int(row_i * 100 / self.ratings_df.shape[0])}%")
                row_i += 1
            record = row
            userId, movieId, r = record["userId"], record["movieId"], record["rating"]
            i, j = np.where(self.user_n_to_index == userId), np.where(self.movie_n_to_index == movieId)

            user_i = np.where(self.user_n_to_index == row["userId"])[0]
            item_i = np.where(self.movie_n_to_index == row["movieId"])[0]
            prediction = self.predict(user_i, item_i)
            e = (r - prediction)

            # copy part of user vector as used in updating item vector
            user_bais_i_cpy = self.user_v[i]

            # update all
            self.user_bias[i] += self.gamma * (e - self.l4 * self.user_bias[i])
            self.item_bias[j] += self.gamma * (e - self.l4 * self.item_bias[j])

            self.user_v[i] += self.gamma * (e * self.item_v[j] - self.l4 * self.user_v[i])
            self.item_v[j] += self.gamma * (e * user_bais_i_cpy - self.l4 * self.item_v[j])

    def learnModelParams(self):
        for train_iteration in range(self.iterations):
            print(f"{train_iteration}/{self.iterations}")
            self.SDE()
            predicitons = self.allPredictions()
            regularised_squared_error = sum(predicitons["reg_sqr_err"])
            avg_regularised_squared_error = sum(predicitons["reg_sqr_err"])/predicitons.shape[0]
            print(f"{regularised_squared_error}, {avg_regularised_squared_error}")
        pass


class reduceSize:
    def __init__(self, ratings, min_movie_raings=50, min_user_reviews=10):
        self.whole_df = ratings
        self.min_movie_raings = min_movie_raings
        self.min_user_reviews = min_user_reviews

        self.df_droped_users = self.userDrop(self.whole_df)
        self.df_droped_movies = self.movieDrop(self.whole_df)

        self.df_droped_movies_users = self.movieDrop(self.df_droped_users)

    def userDrop(self, df):
        count = df['userId'].value_counts()
        invalid = list(count.loc[count < self.min_movie_raings].index)
        return df.loc[df['userId'].isin(invalid)]

    def movieDrop(self, df):
        count = df['movieId'].value_counts()
        invalid = list(count.loc[count < self.min_user_reviews].index)
        return df.loc[df['userId'].isin(invalid)]


def main():
    ml_ratings = "data/ml-25m/ratings.csv"
    ratings = reduceSize(getCSV(ml_ratings), min_movie_raings=50, min_user_reviews=100).df_droped_movies_users
    mat = MatrixFact(ratings, iterations=10, latent_vec_size=10, test_proportion=0.2, l4=0.02, gamma=0.005, verbose=True)


if __name__ == '__main__':
    main()
