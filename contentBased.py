from sklearn.metrics.pairwise import linear_kernel
from data_cleaning import prepareData
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import numpy as np


class ContentCompare:
    def __init__(self, item_df):
        self.item_df = item_df
        self.cos_sim = None

        self.indexToId = pd.DataFrame(self.item_df.clean_items.index)
        self.indexToId["index"] = self.indexToId.index

    def cosine_compare(self, id1, id2):
        index1 = int(list(self.indexToId[self.indexToId["movieId"] == id1]["index"].values)[0])
        index2 = int(list(self.indexToId[self.indexToId["movieId"] == id2]["index"].values)[0])

        if self.cos_sim == None:
            self.cos_sim = pd.DataFrame(cosine_similarity(self.item_df.clean_items.to_numpy()))

        return self.cos_sim[index1][index2]


if __name__ == '__main__':
    items, users = prepareData(load_stored_data=True)
    comp = ContentCompare(items)
    comp.cosine_compare(1, 1)
