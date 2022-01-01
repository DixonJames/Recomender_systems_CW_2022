from MatrixFactorisation import MatrixFact, factoriseMatrix
from data_cleaning import store, prepareData, UserVec, ItemVec, getCSV
import pickle as pkl
import torch
import torch.nn as nn
import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"device: {device}")

# debug
device = "cpu"


def load(path):
    try:
        with open(path, 'rb') as pickle_file:
            content = pkl.load(pickle_file)
        return content
    except:
        return None


def getExampleData(item_vecs, user_latent_vecs):
    item_v = np.array(item_vecs.iloc[[0]])[0]
    user_v = user_latent_vecs[0]
    return catVec(item_v, user_v)


def catVec(item_v, user_v):
    return torch.cat([torch.from_numpy(item_v).to(device), torch.from_numpy(user_v).to(device)])


class MLPBlock(nn.Module):
    def __init__(self, in_d, out_d):
        super().__init__()
        self.in_d = in_d
        self.out_d = out_d

        self.layer = nn.Sequential(nn.Linear(in_d, out_d), nn.ReLU())

    def forward(self, x):
        return self.layer(x)

class MLP(nn.Module):
    def __init__(self, ex_input):
        super().__init__()
        self.example = ex_input
        self.input_dim = ex_input.shape

        self.layer = nn.Sequential(nn.Flatten(),
                                   MLPBlock(32, 16),
                                   MLPBlock(32, 16),
                                   MLPBlock(16, 8),
                                   MLPBlock(8, 4),
                                   nn.Linear(2, 1))

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
        r = MLPBlock(2, 1).forward(r)

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

                    item_v = np.array(self.item_vecs.iloc[item_i])[0]
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
        for epoch in range(epoch_num):
            print(f"Epoch: {epoch}")
            batches = self.genInteractions()
            loss_arr = np.zeros(0)

            batch_number = 0
            for input_vec, rating in batches:
                print(f"{batch_number}/500")
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

    def query(self, movieId, userId):
        with torch.no_grad():
            user_i = np.where(self.user_n_to_index == userId)[0]
            item_i = np.where(self.movie_n_to_index == movieId)[0]

            user_latent_vec = self.user_latent_vecs[user_i]
            item_vec = self.item_vecs[item_i]


def main(load_mat=True):
    # get starting data
    ml_ratings = "data/ml-25m/ratings.csv"
    if not load_mat:
        factoredMatrix = factoriseMatrix()
    else:
        factoredMatrix = load("data/temp/factoredMatrix.obj")
    user_latent_vecs = factoredMatrix.user_latent_v
    item_latent_vecs = factoredMatrix.item_latent_v
    user_interactions = getCSV(ml_ratings)
    item_data = ItemVec(None, None, None, None, load=True).clean_items

    exampleData = getExampleData(item_vecs=item_data, user_latent_vecs=user_latent_vecs)

    model = MLP(exampleData).to(device)
    NC = NuralCollab(NN_model=model, user_latent_vecs=user_latent_vecs, item_vecs=item_data,
                     interactions=user_interactions)

    NC.train()


if __name__ == '__main__':
    main()
