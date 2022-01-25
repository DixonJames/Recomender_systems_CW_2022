from MatrixFactorisation import MatrixFact, factoriseMatrix
from data_cleaning import store, prepareData, UserVec, ItemVec, getCSV
import pickle as pkl
import torch
import torch.nn as nn
import numpy as np
import pandas as pd

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"device: {device}")

# debug
#device = "cpu"


def load(path):
    try:
        with open(path, 'rb') as pickle_file:
            content = pkl.load(pickle_file)
        return content
    except:
        return None


def getExampleData(item_vecs, user_latent_vecs):
    item_v = np.array(item_vecs.clean_items.iloc[[0]])[0]
    user_v = user_latent_vecs[0]
    return catVec(item_v, user_v)


def catVec(item_v, user_v):
    return torch.cat([torch.from_numpy(item_v).to(device), torch.from_numpy(user_v).to(device)])


class MLPBlock(nn.Module):
    def __init__(self, in_d, out_d):
        super().__init__()
        self.in_d = in_d
        self.out_d = out_d

        self.device = device
        self.layer = nn.Sequential(nn.Linear(in_d, out_d), nn.ReLU()).to(self.device)


    def forward(self, x):
        return self.layer(x)


class MLP_network(nn.Module):
    def __init__(self, ex_input, min_score, max_score):
        super().__init__()
        self.example = ex_input
        self.min_score = min_score
        self.max_score = max_score

        self.input_dim = ex_input.shape

        self.final_layer = nn.Sequential(nn.Linear(2, 1)).to(device)

    def block(self, in_dim, out_dim):
        return nn.Sequential(nn.Linear(in_dim, out_dim), nn.ReLU())

    def layers(self, x):
        r = self.block(self.input_dim[0], 2 ** self.largestFactor(), x)
        for p_i in len(self.pows):
            r = self.block(self.pows[-p_i], self.pows[-(p_i + 1)], r)
        return r

    def forward(self, x):
        r = x.view(-1).float()
        r = MLPBlock(self.input_dim[0], 32).forward(r)
        r = MLPBlock(32, 16).forward(r)
        r = MLPBlock(16, 8).forward(r)
        r = MLPBlock(8, 4).forward(r)
        r = MLPBlock(4, 2).forward(r)
        r = self.final_layer.forward(r)
        r = torch.sigmoid(r)

        output = r * (self.max_score - self.min_score + 1) + self.min_score - 0.5

        return output


class NuralCollab:
    def __init__(self, NN_model, user_latent_vecs, item_vecs, interactions):
        self.model = NN_model.to(device)
        self.user_latent_vecs = user_latent_vecs
        self.item_vecs = item_vecs
        self.interactions = interactions
        self.learning_rate = 1e-3

        self.user_n_to_index = self.interactions["userId"].unique()
        self.movie_n_to_index = self.interactions["movieId"].unique()

        self.betas = (0.5, 0.999)
        self.opt = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate, betas=self.betas)

    def genInteractions(self, age=None, country=None, timeRange=None):
        """
        generates interactions based on input filters are for training a model for context based recommendations
        :param age:
        :param country:
        :param timeRange:
        :return:
        """
        for row in self.interactions.iterrows():
            if age is None:
                if country is None:
                    # haevnbt implitmetd timerage yet
                    user_i = np.where(self.user_n_to_index == row[1]["userId"])[0]
                    item_i = np.where(self.movie_n_to_index == row[1]["movieId"])[0]

                    try:
                        item_v = np.array(self.item_vecs.clean_items.iloc[item_i])[0]
                    except:
                        pass

                    user_v = self.user_latent_vecs[user_i][0]

                    input_v = catVec(item_v, user_v)
                    rating = row[1]["rating"]

                    yield input_v, rating

    def constructvector(self, user_latent_vec, item_vec):
        return catVec(item_vec, user_latent_vec)

    def lossFunc(self, prediction: float, rating: float) -> float:
        return prediction - rating

    def exampleData(self):
        row = self.interactions[0]
        user_i = np.where(self.user_n_to_index == row["userId"])[0]
        item_i = np.where(self.movie_n_to_index == row["movieId"])[0]

        user_latent_vec = self.user_latent_vecs[user_i]
        item_vec = self.item_vecs[item_i]

        item_v = self.item_vec.iloc[[item_i]]
        user_v = self.user_latent_vecs[user_i]

        return self.constructvector(user_latent_vec, item_vec)

    def train(self, epoch_num=10):
        row_number = self.interactions.shape[0]
        for epoch in range(epoch_num):
            print(f"Epoch: {epoch}/{epoch_num}")
            batches = self.genInteractions()
            loss_arr = np.zeros(0)

            batch_number = 0
            for input_vec, rating in batches:
                #print(f"{batch_number * 100 / row_number}%")
                rating = torch.from_numpy(np.array([rating]))
                input_vec, rating = input_vec.to(device), rating.to(device)

                prediction = self.model(input_vec)
                # increase loss if says real is fake
                loss = self.lossFunc(prediction, rating)
                self.opt.zero_grad()
                loss.backward(retain_graph=True)
                self.opt.step()

                loss_arr = np.append(loss_arr, loss.item())

                # self.plot(epoch, batch_number, loss_arr, x_real)

                batch_number += 1

        # checkpointModel(self.model, f'score_model_{epoch}.pth')

    def predict(self, movieId, userId):
        with torch.no_grad():
            user_i = np.where(self.user_n_to_index == userId)[0]
            item_i = np.where(self.movie_n_to_index == movieId)[0]

            try:
                user_latent_vec = self.user_latent_vecs[user_i][0]
            except:
                pass
            item_vec = self.item_vecs.clean_items.iloc[item_i].values[0]

            input_v = catVec(item_vec, user_latent_vec)

            input_v = input_v.to(device)

            prediction = self.model(input_v)

        return prediction.numpy()[0]

    def seenPredictions(self, ratings_df=None):
        if ratings_df is None:
            ratings_df = self.interactions

        def pred(row):
            user_i = np.where(self.user_n_to_index == row["userId"])[0]
            item_i = np.where(self.movie_n_to_index == row["movieId"])[0]
            prediction = self.predict(item_i, user_i)

            dct = {"userId": row["userId"],
                   "itemId": row["movieId"],
                   "rating": row["rating"],
                   "prediction": prediction}

            return pd.Series(dct)

        predictions = ratings_df.apply(lambda row: pred(row), axis=1)
        return predictions

    def allPredictions(self, user_id):
        """
                predics all preddicrions for seen and unseen movies for a specific user
                :param user_id:
                :return:
                """
        movie_ids = pd.DataFrame(self.interactions["movieId"].unique())

        def pred(row):
            user_i = user_id
            item_i = row[0]
            prediction = self.predict(item_i, user_i)

            dct = {"userId": user_id,
                   "itemId": row[0],
                   "prediction": prediction}

            return pd.Series(dct)

        predictions = movie_ids.apply(lambda row: pred(row), axis=1)

        return predictions


def neauralCollaberativeModel(load_mat=True, load_model=True, ratings=None):
    # get starting data
    if ratings is None:
        ratings = "data/ml-latest-small/ratings.csv"

    factoredMatrix = factoriseMatrix(load_matrix=load_mat, ratings=ratings)

    if not load_model:
        user_latent_vecs = factoredMatrix.user_latent_v
        item_latent_vecs = factoredMatrix.item_latent_v
        user_interactions = ratings

        item_data, user_data = prepareData(load_stored_data=True)

        exampleData = getExampleData(item_vecs=item_data, user_latent_vecs=user_latent_vecs)
        model = MLP_network(exampleData, 1, 5).to(device)
        NC = NuralCollab(NN_model=model, user_latent_vecs=user_latent_vecs, item_vecs=item_data,
                         interactions=user_interactions)

        NC.train(epoch_num=100)
        store(NC, "data/temp/MLP_100.obj")
    else:
        NC = load("data/temp/MLP_10.obj")

    return NC


if __name__ == '__main__':
    neauralCollaberativeModel(load_mat=False)
