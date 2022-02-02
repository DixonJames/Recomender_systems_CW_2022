import pandas as pd
import numpy as np

from sklearn.metrics.pairwise import cosine_similarity
from data_cleaning import prepareData
from contentBased import ContentCompare
from data_cleaning import load
from MatrixFactorisation import factoriseMatrix, MatrixFact


class Mf_Collaberative_hybrid:
    """
    for experiemts pass in the experimental parcial data MF and the users without wheir reviews
    """
    def __init__(self, user_id, MF_model, reviews, user_vector, classed_user_vector, CB_weighting=0.5):
        self.MF_model = MF_model
        self.user_vector = user_vector
        self.classed_user_vector = classed_user_vector
        self.user_id = user_id

        #reviews should be without the users test reviews
        self.reviews = reviews

        self.cb_weight = CB_weighting
        self.MF_weighting = 1 - CB_weighting

        self.CB_pred = self.contentBasedPrediction()
        self.MF_pred = self.matrixFactorPrediction()

    def contentBasedPrediction(self):
        content_based = ContentCompare(user_v=self.user_vector, class_user_v=self.classed_user_vector)
        all_predictions = content_based.queryUser(profile_vector_type="weighted", distance_measure="cosine")
        return all_predictions

    def matrixFactorPrediction(self):
        if self.MF_model is None:
            self.MF_model = factoriseMatrix(load_matrix=False, iterations=30)

        all_predictions = self.MF_model.allPredictions(self.user_id)
        return all_predictions

    def getCandidates(self, number_candidates=30):
        consumed_content = self.reviews[self.reviews["userId"] == self.user_id]
        CB_candidates = list(
            self.CB_pred[~self.CB_pred["movieId"].isin(consumed_content)].sort_values("prediction")["movieId"].values())
        MF_candidates = list(
            self.MF_pred[~self.MF_pred["movieId"].isin(consumed_content)].sort_values("prediction")["movieId"].values())

        candaite_ids = []
        turn = 0
        while len(candaite_ids) <= number_candidates:
            empty = 0
            if turn == 0:
                if len(CB_candidates) != 0:
                    top_c = CB_candidates.pop()
                    while top_c in set(candaite_ids) and len(CB_candidates) != 0:
                        top_c = CB_candidates.pop()
                    candaite_ids.append(top_c)
                empty += 1
                turn = 1

            else:
                if len(MF_candidates) != 0:
                    top_c = MF_candidates.pop()
                    while top_c in set(candaite_ids) and len(MF_candidates) != 0:
                        top_c = MF_candidates.pop()
                    candaite_ids.append(top_c)
                empty += 1
                turn = 0

            if empty == 2:
                return candaite_ids

        return candaite_ids

    def rankCandidates(self):
        """
        produces overall rank for candidates
        run for list of recomnedations in roder
        :return:
        """
        candiadtes = self.getCandidates(number_candidates=30)

        CB_candidates = self.CB_pred[self.CB_pred["movieId"].isin(candiadtes)].sort_values("prediction")
        CB_candidates.rename({"prediction": "CB_prediction"}, axis='columns')
        MF_candidates = self.MF_pred[self.MF_pred["movieId"].isin(candiadtes)].sort_values("prediction")
        MF_candidates.rename({"prediction": "MF_prediction"}, axis='columns')

        CB_candidates['CB_rank'] = CB_candidates['prediction'].rank(na_option='bottom')
        MF_candidates['MF_rank'] = MF_candidates['prediction'].rank(na_option='bottom')

        combined_df = pd.merge(CB_candidates, MF_candidates, on="movieId", how="outer")

        combined_df["overall_rank"] = self.cb_weight * combined_df["CB_rank"] + self.MF_weighting * combined_df["MF_rank"]
        combined_df.sort_values("overall_rank", ascending=False)
        return combined_df

    def allPredictions(self):
        """
        predics all preddicrions for seen and unseen movies for a specific user.
        use for experiments
        :param user_id:
        :return:
        """

        CB_candidates = self.CB_pred[self.CB_pred["movieId"]].sort_values("prediction")["movieId"].values()
        CB_candidates["prediction"] = CB_candidates["prediction"]*5
        CB_candidates.rename({"prediction": "CB_prediction"}, axis='columns')
        MF_candidates = self.MF_pred[self.MF_pred["movieId"]].sort_values("prediction")["movieId"].values()
        MF_candidates.rename({"prediction": "MF_prediction"}, axis='columns')

        combined_df = pd.merge(CB_candidates, MF_candidates, on="movieId", how="outer")

        combined_df["prediction"] = self.cb_weight * combined_df["CB_prediction"] + self.MF_weighting * combined_df[
            "MF_prediction"]
        combined_df.sort_values("prediction", ascending=False)

        return combined_df





