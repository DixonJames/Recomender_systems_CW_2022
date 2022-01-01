from MatrixFactorisation import MatrixFact, factoriseMatrix
from data_cleaning import store, prepareData, UserVec, ItemVec, getCSV
import pickle as pkl
import torch
import torch.nn as nn
import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"device: {device}")


def load(path):
    try:
        with open(path, 'rb') as pickle_file:
            content = pkl.load(pickle_file)
        return content
    except:
        return None


class MLP(nn.Module):
    def __init__(self, ex_input):
        super().__init__()

    def block(self, in_dim, out_dim, x):
        r = nn.Linear(in_dim, out_dim).forward(x)
        return nn.ReLU().forward(r)

    def forward(self, x):
        r = self.block(32 * 32 * 3, 64, x)
        r = self.block(64, 32, r)
        r = self.block(32, 16, r)
        r = self.block(8, 4, r)
        r = self.block(2, 1, r)

        return r


class NuralCollab:
    def __init__(self, NN_model, user_latent_vecs, item_vecs, interactions):
        self.model = NN_model
        self.user_latent_vecs = user_latent_vecs
        self.item_vecs = item_vecs
        self.interactions = interactions
        self.learning_rate = 2e-4

        self.user_n_to_index = self.interactions["userId"].unique()
        self.movie_n_to_index = self.interactions["movieId"].unique()

        self.opt = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

    def genInteractions(self, age=None, country=None, timeRange=None):
        """
        generates interactions based on input filters are for training a model for context based recommendations
        :param age:
        :param country:
        :param timeRange:
        :return:
        """
        for row in self.interactions.iterrows():
            if row["age"] in range(age - 5, age + 5) or age is None:
                if row["country"] == country or country is None:
                    # haevnbt implitmetd timerage yet
                    user_i = np.where(self.user_n_to_index == row["userId"])[0]
                    item_i = np.where(self.movie_n_to_index == row["movieId"])[0]

                    user_latent_vec = self.user_latent_vecs[user_i]
                    item_vec = self.item_vecs[item_i]
                    rating = row["rating"]

                    yield user_latent_vec, item_vec, rating

    def lossFunc(self, prediction: float, rating: float) -> float:
        return prediction - rating

    def train(self, epoch_num=10, ):
        for epoch in range(epoch_num):
            print(f"Epoch: {epoch}")
            batches = self.genInteractions()
            loss_arr = np.zeros(0)

            batch_number = 0
            for user_latent_vec, item_vec, rating in batches:
                print(f"{batch_number}/500")
                user_latent_vec, item_vec, rating = user_latent_vec.to(device), item_vec.to(device), rating.to(device)

                query_vec = torch.cat(user_latent_vec, item_vec)

                # increase loss if says real is fake
                loss = self.lossFunc(rating)
                self.opt.zero_grad()
                loss.backward(retain_graph=True)
                self.opt.step()

                loss_arr = np.append(loss_arr, loss.item())

                # self.plot(epoch, batch_number, loss_arr, x_real)

                batch_number += 1

        # checkpointModel(self.model, f'score_model_{epoch}.pth')

    def query(self, movieId, userId):
        with torch.no_grad():
            pass


def main(load_mat=True):
    # get starting data
    ml_ratings = "data/ml-25m/ratings.csv"
    if not load_mat:
        factoredMatrix = factoriseMatrix()
    else:
        factoredMatrix = load("data/temp/factoredMatrix.obj")
    user_latent_vecs = factoredMatrix.user_latent_v
    item_latent_vecs = factoredMatrix.item_latent_v
    user_interactions = UserVec(getCSV(ml_ratings)).user_df
    item_data = ItemVec(None, None, None, None, load=True).clean_items

    model = MLP(1)
    NC = NuralCollab(NN_model=model, user_latent_vecs=user_latent_vecs, item_vecs=item_data, interactions=user_interactions)


if __name__ == '__main__':
    main()
