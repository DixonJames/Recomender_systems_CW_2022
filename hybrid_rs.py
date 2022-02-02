import pandas as pd
import numpy as np
from contentBased import ContentCompare
from MatrixFactorisation import factoriseMatrix


class Mf_Collaberative_hybrid:
    """
    for experiemts pass in the experimental parcial data MF and the users without wheir reviews
    """

    def __init__(self, CB_pred, MF_pred, reviews, CB_weighting=0.5):
        # reviews should be without the users test reviews
        self.reviews = reviews

        self.cb_weight = CB_weighting
        self.MF_weighting = 1 - CB_weighting

        self.CB_pred = CB_pred
        self.MF_pred = MF_pred

    def getCandidates(self, number_candidates=30):
        consumed_content = self.reviews
        CB_candidates = list(
            self.CB_pred[~self.CB_pred["itemId"].isin(consumed_content)].sort_values("prediction_collaborative")["itemId"].values)
        MF_candidates = list(
            self.MF_pred[~self.MF_pred["itemId"].isin(consumed_content)].sort_values("prediction_matrix")["itemId"].values)

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

    def rankCandidates(self, number_recs=30):
        """
        produces overall rank for candidates
        run for list of recomnedations in roder
        :return:
        """
        candiadtes = self.getCandidates(number_candidates=number_recs)

        CB_candidates = self.CB_pred[self.CB_pred["itemId"].isin(candiadtes)].sort_values("prediction_collaborative")
        MF_candidates = self.MF_pred[self.MF_pred["itemId"].isin(candiadtes)].sort_values("prediction_matrix")

        CB_candidates['CB_rank'] = CB_candidates['prediction_collaborative'].rank(na_option='bottom')
        MF_candidates['MF_rank'] = MF_candidates['prediction_matrix'].rank(na_option='bottom')

        combined_df = pd.merge(CB_candidates, MF_candidates, on="itemId", how="outer")

        combined_df["overall_rank"] = self.cb_weight * combined_df["CB_rank"] + self.MF_weighting * combined_df[
            "MF_rank"]
        combined_df.sort_values("overall_rank", ascending=False)
        return combined_df

    def allPredictions(self):
        """
        predics all preddicrions for seen and unseen movies for a specific user.
        use for experiments
        :param user_id:
        :return:
        """

        CB_candidates = self.CB_pred.sort_values("prediction_collaborative")
        CB_candidates["prediction_collaborative"] = CB_candidates["prediction_collaborative"] * 5
        MF_candidates = self.MF_pred.sort_values("prediction_matrix")

        combined_df = pd.merge(CB_candidates, MF_candidates, on="itemId", how="outer")

        combined_df["prediction_hybrid"] = self.cb_weight * combined_df["prediction_collaborative"] + self.MF_weighting * \
                                    combined_df[
                                        "prediction_matrix"]
        combined_df.sort_values("prediction_hybrid", ascending=False)

        return combined_df
