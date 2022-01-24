from sklearn.metrics.pairwise import linear_kernel
from data_cleaning import prepareData
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import numpy as np
from scipy.stats.stats import pearsonr



class ContentCompare:
    def __init__(self, item_df, user):
        self.item_df = item_df
        self.user_obj = user
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

    def queryUser(self, profile_vector_type="weighted", distance_measure="cosine"):
        """
        gets a list of the movies that the users user vector
        can do two types,  a single weitted vecetor or 5 vectors of the different rating classes

        :param profile_vector_type: "weighted", "classed"
        :param distance_measure: "cosine", "euclidian", "pearson"
        :return: list of movies with their probs of liking
        """

        movie_df_cpy = self.item_df.clean_items.copy()

        if profile_vector_type == "weighted":
            user_vec = self.user_obj.createUserVector()
            return self.compareVectorToItems(distance_measure=distance_measure, test_vector=user_vec)

        if profile_vector_type == "classed":
            user_vec_classes, user_weight_classes = self.user_obj.createClassUserVecotor()
            class_ratings_df = pd.DataFrame({"movieId": movie_df_cpy.index})

            rating_val = 1
            for rating_vector in user_vec_classes:
                cls_rating = self.compareVectorToItems(distance_measure=distance_measure,
                                                       test_vector=rating_vector).rename(
                    {"comparison_val": f"rating_{rating_val}_similarity"}, axis='columns')
                class_ratings_df = pd.merge(class_ratings_df, cls_rating, on='movieId')
                rating_val += 1

            def mostSimilarRating(row):
                final_scores = [row[f"rating_{i + 1}_similarity"] for i in range(5)]
                top_voted_rating = final_scores.index(max(final_scores)) + 1
                return top_voted_rating / 5

            class_ratings_df["voted_score"] = class_ratings_df.apply(lambda row: mostSimilarRating(row), axis=1)

            return class_ratings_df

        # add user vecs to the movie vecs


if __name__ == '__main__':
    items, users = prepareData(load_stored_data=True)

    comp = ContentCompare(items, users)
    comp.queryUser(1, profile_vector_type="classed", distance_measure="pearson")
