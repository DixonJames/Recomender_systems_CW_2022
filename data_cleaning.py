import pandas as pd
import numpy as np
from functools import cache
from itertools import chain, groupby
from collections import defaultdict
import nltk
from nltk.corpus import stopwords
from sklearn import preprocessing
from itertools import combinations
from sklearn.feature_extraction.text import TfidfVectorizer
from pandas import DataFrame
from scipy import spatial
import gensim
import re
import csv
import pickle as pkl
from random import shuffle
import matplotlib.pyplot as plt


def plot(point_series, names, logScale=False):
    from math import log
    for points, name in zip(point_series, names):
        if logScale:
            points = [(x, log(y)) for x, y in points]
        plt.plot(*zip(*points), '-o', label=name)
    plt.xlabel('Data Train Iteration')
    plt.ylabel('Error Size')
    plt.legend()
    plt.show()

def updateNltkWords():
    try:
        print("downloading common wordlisml_tags_lookupts")
        nltk.download('stopwords')
        nltk.download('punkt')
    except:
        print("couldn't download latest lists")


def getCSV(path, sep=None):
    if sep is None:
        return pd.read_csv(filepath_or_buffer=path, on_bad_lines='skip')
    return pd.read_csv(path, on_bad_lines='skip', sep=sep)


def store(o, path):
    filehandler = open(f"{path}", "wb")
    pkl.dump(o, filehandler)


def load(path):
    try:
        with open(path, 'rb') as pickle_file:
            content = pkl.load(pickle_file)
        return content
    except:
        return None


class reduceSize:
    def __init__(self, items, users, min_movie_raings=50, min_user_reviews=10):
        self.items, self.users = items, users
        self.min_movie_raings = min_movie_raings
        self.min_user_reviews = min_user_reviews

    def invalidUsers(self, df):
        count = df['userId'].value_counts()
        invalid = list(count.loc[count < self.min_movie_raings].index)
        return invalid

    def invalidMovies(self, df):
        count = df['movieId'].value_counts()
        invalid = list(count.loc[count < self.min_user_reviews].index)
        return invalid

    def reduced(self):
        invalid_movies = self.invalidMovies(self.users.ratings)
        invalid_users = self.invalidUsers(self.users.ratings)

        self.users.ratings = self.users.ratings[~self.users.ratings["userId"].isin(invalid_users)]
        self.users.ratings = self.users.ratings[~self.users.ratings["movieId"].isin(invalid_movies)]

        self.users.user_df = self.users.user_df[~self.users.user_df["userId"].isin(invalid_users)]

        self.items.clean_items = self.items.clean_items[~self.items.clean_items.index.isin(invalid_movies)]
        self.items.genres = self.items.genres[~self.items.genres["movieId"].isin(invalid_movies)]
        self.items.tags = self.items.tags[~self.items.tags["movieId"].isin(invalid_movies)]

        return self.users, self.items


class Parseplots:
    def __init__(self, path, load=False):
        """
        :param path: path of ftp.fu-berlin.de/pub/misc/movies/database/frozendata/plot.list
        """
        self.path = path
        self.EOSection = "-------------------------------------------------------------------------------\n"

        if not load:
            self.df = self.createDF()
            self.save()
        else:
            self.df = self.load()

    def save(self):
        store(self.df, "data/temp/plots.pkl")

    def load(self):
        return load("data/temp/plots.pkl")

    def lineGen(self):
        with open(self.path, 'rb') as file:
            for l in file.readlines():
                try:
                    yield l.decode("utf-8")
                except:
                    pass

    def sectionGen(self):
        lines = self.lineGen()
        section = []

        for line in lines:
            if set(line) != set(self.EOSection):
                section.append(line)
            else:
                yield section
                section = []

    def partsGen(self):
        sections = self.sectionGen()
        for section in sections:
            # print(section)
            balnc_indices = [index for index, element in enumerate(section) if element == "\n"]

            try:
                title = section[0]
                plot = section[balnc_indices[0]:balnc_indices[max(1, 0)]]
                yield " ".join(plot).replace("\n", " ").replace("PL:", " "), re.search('"(.*)"', title).group(
                    1).replace("#", "").lower()
            except:
                continue

    def createDF(self):
        plotGen = self.partsGen()
        plot_df = pd.DataFrame(columns=["title", "plot"])
        for plot, title in plotGen:
            plot_df = plot_df.append({"title": title, "plot": plot}, ignore_index=True)

        return plot_df


class Doc2VecSimilarity:
    """
    for creating doc vectors in relation to a corpus of docs
    """

    def __init__(self, corpus, vec_size=50):
        self.training_corpus = self.preProcesCorpus(corpus)

        self.vec_size = vec_size
        self.model = self.doc2VecPipeline()

    def preProcesCorpus(self, corpus):
        """
        removes stopwords
        and words appearing only one time
        :param corpus:
        :return:
        """
        s = set(stopwords.words('english'))
        working_corpus = [[word for word in article.split(" ") if word not in s] for article in corpus]
        clean_corpus = [" ".join(article) for article in working_corpus]

        word_freq = defaultdict(int)

        for aritcle in working_corpus:
            for word in aritcle:
                word_freq[word] += 1

        index = 0

        final_corpus = [[word for word in aritcle if word_freq[word] > 1] for aritcle in working_corpus]

        return final_corpus

    def addTokens(self, doc, index):
        """
        tokenizes a document into list of words
        :param doc: big docuemtb string
        :param index:
        :return: list of words
        """
        stringver = ""
        tokens = gensim.utils.simple_preprocess(' '.join(doc))
        return gensim.models.doc2vec.TaggedDocument(tokens, [index])

    def doc2VecVocab(self, corpai):
        """
        trainds the doc2vec model with the corpus of documents
        :param corpai:
        :return:
        """
        training_corpai = []
        index = 0
        for doc in corpai:
            training_corpai.append(self.addTokens(doc, index))
            index += 1

        model = gensim.models.doc2vec.Doc2Vec(vector_size=self.vec_size, min_count=2, epochs=40)
        model.build_vocab(training_corpai)
        return model, training_corpai

    def doc2VecPipeline(self):
        """
        main structure for training  and testing the doc2vec model
        :return: train doc2vec model
        """
        corpus = self.training_corpus

        corpus = list(corpus for corpus, _ in groupby(corpus))

        model, training_corpai = self.doc2VecVocab(corpus)
        model.train(training_corpai, total_examples=model.corpus_count, epochs=model.epochs)
        return model

    @cache
    def cosineSimilarity(self, a_vec, b_vec):
        """
        computed cosine similarity between two vectors
        :param a_vec:
        :param b_vec:
        :return:
        """
        similarity = 1 - spatial.distance.cosine(a_vec, b_vec)

        return similarity

    @cache
    def queryDoc2VecModel(self, articleString):
        """
        queries the trained model with a string
        :param articleString: query string
        :return: vectorisation of said input string
        """
        return self.model.infer_vector(gensim.utils.simple_preprocess(articleString))


class ItemVec:
    def __init__(self, tags, tag_labels, genres, plots, load=False, tag_vectoriser=None, plot_vectoriser=None):
        """

        :param plots: df of movie plots
        :param tags: df of movie tags
        :param tag_labels: df relating tag id to string tag vlaues
        :param genres: df containgin genra for each movie
        :param load: boolen value that indicates wether to load in pre-computed values
        :param tag_vectoriser: modle that vectorisese tag stings
        :param plot_vectoriser: model that vectorises plot stings
        """
        updateNltkWords()
        self.tag_lookup = tag_labels
        self.plot_vectoriser = plot_vectoriser
        self.tag_vectoriser = tag_vectoriser
        self.tv_len = 10

        if not load:
            """
            will go though and process fresh files
            TAKES A WHILE!
            """
            self.plots = plots
            self.tags = tags
            self.genres = genres

            # to create the modified DF
            # self.genres_Tf_IFD()

            self.insertTagsPlots()
            self.insertMovieVector()
            self.save()

        else:
            self.plots = None
            self.tags = None
            self.genres = None

            self.load()

        self.clean_items = self.cleanUpDf()

    def MinMaxScaleColl(self, coll):
        vals = coll.values  # returns a numpy array
        scaled = preprocessing.MinMaxScaler().fit_transform(vals.reshape(-1, 1)).T[0]
        return pd.Series(scaled)

    def cleanUpDf(self):
        """
        removes df with unessusary collumbs removed from genras DF
        :return:
        """
        clean = self.genres.copy()
        year = pd.Series

        def extractYear(row):
            title = row["title"].replace(" ", "")
            try:
                return int(title[-5:-1])
            except:
                return 2000

        years = self.MinMaxScaleColl(clean.apply(lambda row: extractYear(row), axis=1))
        clean = clean.assign(year=years)

        for coll in [c for c in list(clean.axes[1].array) if "tag_vec" in c]:
            clean[coll] = self.MinMaxScaleColl(clean[coll])

        clean.index = clean["movieId"]
        clean.drop(
            ["compressed_title", "movie_string", "genres_x", "genres_y", "title", "top_tags", "plots", "movieId"],
            axis=1, inplace=True)
        return clean.replace({np.NAN: 0.0})

    def cleanPlots(self):
        """
        for time being unused
        :return:
        """
        plots = self.plots

        def getTagVec(row):
            return pd.Series(self.tag_vectoriser.queryDoc2VecModel(row['top_tags']))

        vecs = self.genres.apply(lambda row: getTagVec(row), axis=1)
        self.genres[[f"tag_vec_{i}" for i in range(self.tv_len)]] = vecs
        self.save()

    def save(self):
        # store(self.plots, "data/temp/plots.pkl")
        store(self.tags, "data/temp/tags.pkl")
        store(self.genres, "data/temp/genres.pkl")

    def load(self):
        # self.plots = load("data/temp/plots.pkl")
        self.tags = load("data/temp/tags.pkl")
        self.genres = load("data/temp/genres.pkl")

    def idTranslate(self, id=None, movietitle=None):
        "tranlates between ID and movie title"
        if id is not None:
            return list(self.genres.loc[self.genres['movieId'] == id]["title"])[0]
        if movietitle is not None:
            return list(self.genres.loc[self.genres['title'] == movietitle]["movieId"])[0]

    def insertTagsPlots(self):
        """
        get top tags for each of the records and create a vector out of them
        :return:
        """
        self.genres["top_tags"] = np.zeros(self.genres.shape[0])

        last = np.array([])

        def regulariseTitle(title):
            return re.sub("[\(\[].*?[\)\]]", "", title).replace(" ", "").lower()

        self.genres["compressed_title"] = pd.Series(np.vectorize(regulariseTitle)(self.genres["title"]))
        self.plots["compressed_title"] = pd.Series(np.vectorize(regulariseTitle)(self.plots["title"]))

        def get_tag_str(id):
            return " ".join(self.tags[self.tags["movieId"] == id]["tag"].values)

        def get_plot_str(row):
            title = row["compressed_title"]
            return " ".join(self.plots[self.plots["compressed_title"] == title]["plot"].values)

        self.genres["plots"] = self.genres.apply(lambda row: get_plot_str(row), axis=1)

        v_get_tag_str = np.vectorize(get_tag_str)
        self.genres["top_tags"] = v_get_tag_str(self.genres["movieId"])

        self.save()

    def insertMovieVector(self):

        def regulariseTitle(title):
            return re.sub("[\(\[].*?[\)\]]", "", title).lower()

        self.genres["movie_string"] = pd.Series(np.vectorize(regulariseTitle)(self.genres["title"])) + self.genres[
            "top_tags"] + self.genres["plots"]
        corpus = self.genres['movie_string'].values

        if self.tag_vectoriser is None:
            self.tag_vectoriser = Doc2VecSimilarity(corpus, vec_size=self.tv_len)
            store(self.tag_vectoriser, "data/temp/tag_vectorise")

        for i in range(self.tv_len):
            self.genres[f"movie_str_vec_{i}"] = np.zeros(self.genres.shape[0])

        def getTagVec(row):
            return pd.Series(self.tag_vectoriser.queryDoc2VecModel(row['movie_string']))

        vecs = self.genres.apply(lambda row: getTagVec(row), axis=1)
        self.genres[[f"movie_str_vec_{i}" for i in range(self.tv_len)]] = vecs
        self.save()

    def genres_Tf_IFD(self):
        """
        creates boolean coll for each genre
        """
        genres = "Action* Adventure* Animation* Children* Comedy* Crime* Documentary* Drama* Fantasy* Film-Noir* Horror* Musical* Mystery* Romance* Sci-Fi* Thriller* War* Western* (no genres listed)"
        genres = [g.replace(" ", "") for g in genres.split("*")]
        genres_str = ' '.join(genres)
        a = [c for i in range(1, 4) for c in combinations(genres_str.split(), r=i)]

        tf = TfidfVectorizer(stop_words='english')
        tfidf_matrix = tf.fit_transform(self.genres["genres"])
        tf_idf_df = pd.DataFrame(tfidf_matrix.todense(), columns=tf.get_feature_names_out(), index=self.genres["title"])
        tf_idf_df.reset_index(inplace=True)

        self.genres = pd.merge(self.genres, tf_idf_df, on="title")


class UserVec:
    def __init__(self, ratingsMat):
        self.ratings = ratingsMat
        self.user_num = self.ratings["userId"].unique().shape[0]

        self.means = self.userMeanRating()
        self.reviews = self.userNumberRatings()
        self.ages = pd.Series(np.random.choice(80, self.user_num))
        self.country = pd.Series(np.random.choice(195, self.user_num))

        self.user_df = self.createUserVec()

    def MinMaxScaleColl(self, coll):
        vals = coll.values  # returns a numpy array
        scaled = preprocessing.MinMaxScaler().fit_transform(vals.reshape(-1, 1)).T[0]
        return pd.Series(scaled)

    def userMeanRating(self):
        means = self.ratings.groupby("userId")["rating"].mean()
        return means.append(pd.Series(np.mean(means)))

    def userNumberRatings(self):
        ratings = self.ratings["userId"].value_counts()
        return ratings.append(pd.Series(np.mean(ratings)))

    def createUserVec(self):
        user_df = pd.DataFrame(columns=["userId", "mean_score", "review_num", "age", "country"])
        user_df["userId"] = self.ratings["userId"].unique()
        user_df["mean_score"] = self.MinMaxScaleColl(self.means)
        user_df["review_num"] = self.MinMaxScaleColl(self.reviews)
        user_df["age"] = self.MinMaxScaleColl(self.ages)
        user_df["country"] = self.MinMaxScaleColl(self.country)

        return user_df


def prepareData(load_stored_data=False, reduce=True, min_movie_raings=30, min_user_reviews=100):
    plotParts = "data/plots/IMDB/plot.list"

    ml_genres = "data/ml-latest-small/movies.csv"
    ml_ratings = "data/ml-latest-small/ratings.csv"
    ml_tags = "data/ml-latest-small/tags.csv"

    tag_vec = load("data/temp/tag_vectorise")

    if not load_stored_data:
        plotParts = Parseplots(plotParts, load=True).df
        ml_genres = getCSV(ml_genres)
        ml_tags = getCSV(ml_tags)

        items = ItemVec(tags=ml_tags, tag_labels=ml_tags, genres=ml_genres,
                        tag_vectoriser=None, plots=plotParts)
        users = UserVec(getCSV(ml_ratings))

    else:
        items = ItemVec(None, None, ml_tags, None, load=True, tag_vectoriser=tag_vec)
        users = UserVec(getCSV(ml_ratings))

    # ml_ratings = getCSV("data/ml-25m/ratings.csv")

    if reduce:
        users, items = reduceSize(items, users, min_movie_raings=min_movie_raings,
                                  min_user_reviews=min_user_reviews).reduced()

    return items, users


class DataSets:
    """
    training and testing split on each user
    """

    def __init__(self, items, users, group_number):
        self.items = items
        self.users = users
        self.group_number = group_number

        self.movie_in_group_check = self.createMovieChecklist()
        self.ratingsGroups = self.createRatingsGroups()

        self.reballenceSets()

    def createMovieChecklist(self):
        base_df = pd.DataFrame(data=self.users.ratings["movieId"].unique()).copy()
        overall_df = pd.DataFrame(data=self.users.ratings["movieId"].unique()).copy()
        for group_n in range(self.group_number):
            overall_df[f"group_{group_n}"] = pd.Series([0 for _ in range(base_df.shape[0])])
        return overall_df.rename({0: "movieId"}, axis='columns')

    def splitUser(self, ratings):
        indexes = list(ratings.index)
        shuffle(indexes)
        groups = [ratings[ratings.index.isin(list(ratings.index)[i::self.group_number])] for i in
                  range(self.group_number)]
        shuffle(groups)
        return groups

    def createRatingsGroups(self):
        ratingsgroups = [pd.DataFrame(columns=["userId", "movieId", "rating", "timestamp"]) for _ in
                         range(self.group_number)]
        for user_index in self.users.ratings["userId"].unique():
            all_user_ratings = self.users.ratings[self.users.ratings["userId"] == user_index]
            user_index_groups = self.splitUser(all_user_ratings)

            for i in range(len(ratingsgroups)):
                ratingsgroups[i] = ratingsgroups[i].append(user_index_groups[i])

        return ratingsgroups

    def gen_movie_ballance(self, groups):
        for group_i in range(len(groups)):
            group = self.ratingsGroups[group_i]
            counts = group["movieId"].value_counts().reset_index()
            counts = counts.astype('int')
            counts = counts.rename({"index": "movieId", "movieId": f"group_{group_i}"}, axis='columns')


            for count_record in counts.iterrows():
                self.movie_in_group_check.loc[
                    self.movie_in_group_check["movieId"] == count_record[1]["movieId"], f"group_{group_i}"] = int(
                    count_record[1][f"group_{group_i}"])

        for count_record in self.movie_in_group_check.iterrows():
            for c_i in range(len(self.ratingsGroups)):
                if count_record[1][f"group_{c_i}"] == 0:
                    yield count_record, c_i

    def reballenceSets(self):
        for movieId, group_num in self.gen_movie_ballance(self.ratingsGroups):
            #this is fiiiine
            self.createRatingsGroups()
            break

    def genCrossFoldGroups(self):
        for group_i in range(self.group_number):
            test = self.ratingsGroups[group_i]
            train = pd.concat([self.ratingsGroups[i] for i in range(self.group_number)if i != group_i])
            yield test, train






if __name__ == '__main__':
    items, users = prepareData(load_stored_data=True)

    set_split = DataSets(items, users, 5)
    for test, train in set_split.genCrossFoldGroups():
        print(test, train)
