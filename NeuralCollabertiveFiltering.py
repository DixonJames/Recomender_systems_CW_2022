from MatrixFactorisation import MatrixFact, factoriseMatrix
from data_cleaning import store, prepareData, UserVec, ItemVec, getCSV, plot
import pickle as pkl
import torch
import math
import torch.nn as nn
import numpy as np
import pandas as pd

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"device: {device}")

all_plts = []


# debug
# device = "cpu"


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


def plotEpockGraph():
    res = load("data/temp/midlayer_cahgne_all_plts.obj")
    all_series = []
    series_names = []
    layers = 1
    for series in res:

        if layers % 10 == 0:
            all_series.append(series['ABSE'])
            series_names.append(f"ABSE {layers / 10} NN layers")
        layers += 1
    plot(all_series, series_names)


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
    def __init__(self, ex_input, min_score, max_score, midlayers=4):
        super().__init__()
        self.midlayers = midlayers
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

    def layerChooser(self):
        all_layers = [32, 16, 8, 4, 2]
        if self.midlayers == 4:
            return all_layers

        if self.midlayers == 3:
            return [32, 16, 4, 2]

        if self.midlayers == 2:
            return [32, 4, 2]

        if self.midlayers == 1:
            return [32, 2]

    def forward(self, x):
        r = x.view(-1).float()

        all_layers = [32, 16, 8, 4, 2]

        r = MLPBlock(self.input_dim[0], 32).forward(r)

        midlayer_dims = self.layerChooser()
        for i in range(len(midlayer_dims) - 1):
            r = MLPBlock(midlayer_dims[i], midlayer_dims[i + 1]).forward(r)

        r = self.final_layer.forward(r)
        r = torch.sigmoid(r)

        output = r * (self.max_score - self.min_score + 1) + self.min_score - 0.5

        return output


class NuralCollab:
    def __init__(self, user_latent_vecs, item_vecs, interactions, train_test_split=None, model_midlayers=4):

        exampleData = getExampleData(item_vecs=item_vecs, user_latent_vecs=user_latent_vecs)
        self.model = MLP_network(exampleData, 1, 5, midlayers=model_midlayers).to(device)

        self.user_latent_vecs = user_latent_vecs
        self.item_vecs = item_vecs
        self.interactions = interactions

        if train_test_split:
            self.train_test_sets = load("data/temp/experiments/split_sets.DataSets.obj")
        else:
            self.train_test_sets = None

        self.learning_rate = 1e-3

        self.user_n_to_index = self.interactions["userId"].unique()
        self.movie_n_to_index = self.interactions["movieId"].unique()

        self.betas = (0.5, 0.999)
        self.opt = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate, betas=self.betas)

    def extractbatchinfo(self, batch):
        input_vs = []
        for row in batch.iterrows():
            user_i = np.where(self.user_n_to_index == row[1]["userId"])[0]
            item_i = np.where(self.movie_n_to_index == row[1]["movieId"])[0]

            try:
                item_v = np.array(self.item_vecs.clean_items.iloc[item_i])[0]
                user_v = self.user_latent_vecs[user_i][0]
                input_v = catVec(item_v, user_v)

                input_vs.append(input_v.to(device))
            except:
                continue

        ratings = [torch.tensor([i]).to(device) for i in batch["rating"].values]

        return input_vs, ratings

    def genInteractions(self, alt_interactions=None, age=None, country=None, timeRange=None):
        """
        generates interactions based on input filters are for training a model for context based recommendations
        :param age:
        :param country:
        :param timeRange:
        :return:
        """
        interactions = self.interactions
        if alt_interactions is not None:
            interactions = alt_interactions
        interactions = interactions.sample(frac=1, random_state=1).reset_index(drop=True)

        for row_i in range(interactions.shape[0] % 256):
            minibatch = interactions.sample(n=256, random_state=1, replace=False)
            yield self.extractbatchinfo(minibatch)

        minibatch = interactions.sample(frac=1, random_state=1, replace=False)
        yield self.extractbatchinfo(minibatch)

    def constructvector(self, user_latent_vec, item_vec):
        return catVec(item_vec, user_latent_vec)

    def lossFunc(self, prediction: float, rating: float) -> float:
        return rating - prediction

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

        if self.train_test_sets is not None:
            test, train = next(self.train_test_sets.genCrossFoldGroups())

        RMSE_points = []
        ABSE_points = []
        for epoch in range(epoch_num):
            print(f"Epoch: {epoch + 1}/{epoch_num}")

            if self.train_test_sets is None:
                batches = self.genInteractions()
            else:
                batches = self.genInteractions(alt_interactions=train)

            loss_arr = np.zeros(0)

            for input_vecs, ratings in batches:
                # print(f"{batch_number * 100 / row_number}%")
                loss = []
                for input_vec, rating in zip(input_vecs, ratings):
                    input_vec, rating = input_vec.to(device), rating.to(device)

                    prediction = self.model(input_vec)
                    # increase loss if says real is fake

                    loss.append(self.lossFunc(prediction, rating))

                    # self.plot(epoch, batch_number, loss_arr, x_real)

                loss = torch.mean(torch.stack(loss))
                self.opt.zero_grad()
                loss.backward(retain_graph=True)
                self.opt.step()

            if self.train_test_sets is not None:
                predicitons = self.seenPredictions(ratings_df=test)

                regularised_squared_error = math.sqrt(
                    sum((predicitons["rating"] - predicitons["prediction"]) ** 2) / predicitons.shape[0])
                mean_abs_error = sum(abs(predicitons["rating"] - predicitons["prediction"])) / predicitons.shape[0]
                print(f"RMSE:{regularised_squared_error}, ABS_err:{mean_abs_error}")
                RMSE_points.append((epoch, regularised_squared_error))
                ABSE_points.append((epoch, mean_abs_error))

                # if epoch % 10 == 0:
                # plot(point_series=[RMSE_points, ABSE_points], names=["RMSE", "ABS_Error"])
                tot_points = {"RMSE": RMSE_points, "ABSE": ABSE_points}
                all_plts.append(tot_points)

        pass
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

        return prediction.to("cpu").numpy()[0]

    def seenPredictions(self, ratings_df=None):
        if ratings_df is None:
            ratings_df = self.interactions

        def pred(row):
            user_i = row["userId"]
            item_i = row["movieId"]
            with torch.no_grad():
                prediction = self.predict(item_i, user_i)

            dct = {"userId": row["userId"],
                   "itemId": row["movieId"],
                   "rating": row["rating"],
                   "prediction": prediction}

            return pd.Series(dct)

        predictions = ratings_df.apply(lambda row: pred(row), axis=1)
        return predictions

    def allPredictions(self, user_id, alt_df=None):
        """
                predics all preddicrions for seen and unseen movies for a specific user
                :param user_id:
                :return:
                """
        if alt_df is not None:
            interactions = alt_df
        else:
            interactions = self.interactions

        movie_ids = pd.DataFrame(interactions["movieId"].unique())

        def pred(row):
            user_i = user_id
            item_i = row[0]
            prediction = self.predict(item_i, user_i)

            dct = {"userId": user_id,
                   "itemId": row[0],
                   "prediction_NCF": prediction}

            return pd.Series(dct)

        predictions = movie_ids.apply(lambda row: pred(row), axis=1)

        return predictions


def neauralCollaberativeModel(load_mat=True, pass_mat=None, load_model=True, epoch_num=3, model_midlayers=4,
                              train_test_split=False, ratings=None):
    # get starting data
    item_data, user_data = prepareData(load_stored_data=True, reduce=True, min_user_reviews=10, min_movie_raings=50)
    if ratings is not None:
        user_data.ratings = ratings

    if pass_mat is None:
        factoredMatrix = factoriseMatrix(load_matrix=load_mat, ratings=user_data.ratings, iterations=30)
    else:
        factoredMatrix = pass_mat

    if not load_model:
        user_latent_vecs = factoredMatrix.user_latent_v
        exampleData = getExampleData(item_vecs=item_data, user_latent_vecs=user_latent_vecs)
        # model = MLP_network(exampleData, 1, 5, midlayers=model_midlayers).to(device)

        NC = NuralCollab(user_latent_vecs=user_latent_vecs, item_vecs=item_data,
                         interactions=user_data.ratings, train_test_split=train_test_split,
                         model_midlayers=model_midlayers)
        NC.train(epoch_num=epoch_num)

        #store(NC, "data/temp/main_use/NCM_model_l3_e4.obj")
    else:
        NC = load("data/temp/main_use/NCM_model_l3_e4.obj")

    return NC


if __name__ == '__main__':
    # best number of epocks is 3
    # plot(point_series=[[(1,2), (3,4)], [(2,2), (3,5), (1,6)]], names=["RMSE", "ABS_Error"])

    NC = neauralCollaberativeModel(load_mat=True, load_model=False, model_midlayers=3, epoch_num=4)
    store(NC, "data/temp/main_use/NCM_model_l3_e4.obj")
