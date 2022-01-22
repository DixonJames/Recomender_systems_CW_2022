from sklearn.metrics.pairwise import linear_kernel
from data_cleaning import prepareData
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import numpy as np
from scipy.stats.stats import pearsonr


class User:
    def __init__(self, whole_user_df, movie_df, user_id):
        self.user_id = user_id
        self.user_df = whole_user_df.user_df[whole_user_df.user_df["userId"] == self.user_id]
        self.user_reviews = whole_user_df.ratings[whole_user_df.ratings["userId"] == self.user_id]

        self.user_mean_score_diff = -whole_user_df.user_df["mean_score"].mean() + float(whole_user_df.user_df[whole_user_df.user_df["userId"] == self.user_id]["mean_score"])

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
        user_weight_counts = [(self.user_reviews[self.user_reviews["rating"] == i+1].shape[0]) for i in range(5)]
        user_weight_classes = [1/(1+sorted(user_weight_counts, reverse=True).index(i)) for i in user_weight_counts]

        for index, row in self.user_reviews.iterrows():
            rating = int(row["rating"])
            movie_id = row["movieId"]
            movie_vec = np.array(self.movie_df.iloc[int(movie_id)])

            user_vec_classes[rating-1] += movie_vec

        for user_vec_i in range(len(user_vec_classes)):
            user_vec_classes[user_vec_i] = user_vec_classes[user_vec_i] / user_vec_classes[user_vec_i].shape[0]

        return user_vec_classes, user_weight_classes


class ContentCompare:
    def __init__(self, item_df, user_df):
        self.item_df = item_df
        self.user_df = user_df
        self.cos_sim = None
        self.pearson_sim = None
        self.euclidian_sim = None
        self.indexToId = pd.DataFrame(self.item_df.clean_items.index)
        self.indexToId["index"] = self.indexToId.index

    def pearson_compare(self, id1, id2=None, item_df=None):
        index1 = int(list(self.indexToId[self.indexToId["movieId"] == id1]["index"].values)[0])
        if id2 is not None:
            index2 = int(list(self.indexToId[self.indexToId["movieId"] == id2]["index"].values)[0])


        if item_df is None:
            item_df = self.item_df.clean_items
        self.pearson_sim = pd.DataFrame(np.corrcoef(item_df.to_numpy()))

        if id2 is not None:
            return self.pearson_sim[index1][index2]
        return self.pearson_sim[index1]

    def cosine_compare(self, id1, id2=None, item_df=None):
        index1 = int(list(self.indexToId[self.indexToId["movieId"] == id1]["index"].values)[0])
        if id2 is not None:
            index2 = int(list(self.indexToId[self.indexToId["movieId"] == id2]["index"].values)[0])

        if item_df is None:
            item_df = self.item_df.clean_items
        self.cos_sim = pd.DataFrame(cosine_similarity(item_df.to_numpy()))

        if id2 is not None:
            return self.cos_sim[index1][index2]
        return self.cos_sim[index1]

    def euclidian_compare(self, id1, id2):
        index1 = int(list(self.indexToId[self.indexToId["movieId"] == id1]["index"].values)[0])
        if id2 is not None:
            index2 = int(list(self.indexToId[self.indexToId["movieId"] == id2]["index"].values)[0])

        v1 = self.item_df.clean_items[index1]
        v2 = self.item_df.clean_items[index2]

        return np.linalg.norm(v1 - v2)[0]

    def compareVectorToItems(self, distance_measure, test_vector):
        movie_df_cpy = self.item_df.clean_items.copy()


        user_vec_index = max(np.array(self.item_df.clean_items.index)) + 1
        user_vec = test_vector
        movie_df_cpy.loc[user_vec_index] = user_vec

        self.indexToId.loc[max(np.array(self.indexToId.index)) + 1] = [user_vec_index,
                                                                       max(np.array(self.indexToId.index)) + 1]

        if distance_measure == "cosine":
            movie_similarity_predictions = self.cosine_compare(user_vec_index, item_df=movie_df_cpy)

        if distance_measure == "pearson":
            movie_similarity_predictions = self.pearson_compare(user_vec_index, item_df=movie_df_cpy)

        userprofile_to_movie_similarities = pd.DataFrame(
            {"comparison_val": self.cosine_compare(user_vec_index, item_df=movie_df_cpy),
             "movieId": self.indexToId["movieId"]}).sort_values(by=['comparison_val'], ascending=False)

        userprofile_to_movie_similarities = userprofile_to_movie_similarities[
            userprofile_to_movie_similarities["movieId"] != user_vec_index]

        return userprofile_to_movie_similarities

    def queryUser(self, userId, profile_vector_type="weighted", distance_measure="cosine"):
        """
        gets a list of the movies that the users user vector
        can do two types,  a single weitted vecetor or 5 vectors of the different rating classes
        :param userId:
        :param profile_vector_type: "weighted", "classed"
        :param distance_measure: "cosine", "euclidian", "pearson"
        :return: list of movies with their probs of liking
        """
        test_user = User(self.user_df, self.item_df, userId)
        movie_df_cpy = self.item_df.clean_items.copy()

        if profile_vector_type == "weighted":
            user_vec = test_user.createUserVector()
            return self.compareVectorToItems(distance_measure=distance_measure, test_vector=user_vec)


        if profile_vector_type == "classed":
            user_vec_classes, user_weight_classes = test_user.createClassUserVecotor()
            class_ratings_df = pd.DataFrame({"movieId":movie_df_cpy.index})

            rating_val = 1
            for rating_vector in user_vec_classes:
                cls_rating = self.compareVectorToItems(distance_measure=distance_measure, test_vector=rating_vector).rename({"comparison_val":f"rating_{rating_val}_similarity"}, axis='columns')
                class_ratings_df = pd.merge(class_ratings_df, cls_rating, on='movieId')
                rating_val+=1

            def mostSimilarRating(row):
                final_scores = [row[f"rating_{i+1}_similarity"] for i in range(5)]
                top_voted_rating = final_scores.index(max(final_scores)) + 1
                return top_voted_rating/5

            class_ratings_df["voted_score"] = class_ratings_df.apply(lambda row: mostSimilarRating(row), axis=1)

            return class_ratings_df

        # add user vecs to the movie vecs


if __name__ == '__main__':
    items, users = prepareData(load_stored_data=True)

    comp = ContentCompare(items, users)
    comp.queryUser(1, profile_vector_type="classed", distance_measure="pearson")
