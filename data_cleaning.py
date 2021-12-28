import pandas as pd
import numpy as np
from functools import cache
from itertools import chain, groupby
from collections import defaultdict
import nltk
from nltk.corpus import stopwords
from scipy import spatial
import gensim
import csv
import pickle as pkl



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


class Doc2VecSimilarity:
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
    def __init__(self, plots, tags, tag_labels, genres, load=False, tag_vectoriser=None, plot_vectoriser=None):
        self.tag_lookup = tag_labels
        self.tag_vectoriser = tag_vectoriser
        self.plot_vectoriser = plot_vectoriser

        self.tv_len = 10

        if load == False:
            self.plots = plots
            self.tags = tags
            self.genres = genres

            # to create the modified DF
            self.insertTagWords()
            self.sperateGenres()
        else:
            self.plots = None
            self.tags = None
            self.genres = None

            self.load()

        self.insertTagVector()

    def save(self):
        store(self.plots, "data/temp/plots.pkl")
        store(self.tags, "data/temp/tags.pkl")
        store(self.genres, "data/temp/genres.pkl")

    def load(self):
        self.plots = load("data/temp/plots.pkl")
        self.tags = load("data/temp/tags.pkl")
        self.genres = load("data/temp/genres.pkl")

    def idTranslate(self, id=None, movietitle=None):
        "tranlates between ID and movie title"
        if id is not None:
            return list(self.genres.loc[self.genres['movieId'] == id]["title"])[0]
        if movietitle is not None:
            return list(self.genres.loc[self.genres['title'] == movietitle]["movieId"])[0]

    def topTags(self, id, number=50):
        "returns top tags of the specifies movie"
        rec_n = list(
            self.tags.loc[self.tags['movieId'] == id].sort_values(by='relevance', ascending=False, ignore_index=True)[
                'tagId'][:number])
        rec_w = []
        for rn in rec_n:
            rec_w.append(self.tag_lookup.loc[self.tag_lookup['tagId'] == rn]["tag"].values[0])
        return rec_w

    def insertTagWords(self):
        """
        get top tags for each of the records and create a vector out of them
        :return:
        """
        self.genres["top_tags"] = np.zeros(self.genres.shape[0])
        for index, rec in self.genres.iterrows():
            id = rec["movieId"]
            topTags = self.topTags(id, number=50)
            tag_str = ""

            for t in topTags:
                tag_str += t
                tag_str += " "

            self.genres.loc[index, "top_tags"] = tag_str

            print(index, self.genres.shape[0])
        self.save()

    def insertTagVector(self):
        corpus = self.genres['top_tags'].values

        if self.tag_vectoriser is None:
            self.tag_vectoriser = Doc2VecSimilarity(corpus, vec_size=self.tv_len)
            store(self.tag_vectoriser, "data/temp/tag_vectorise")


        for i in range(self.tv_len):
            self.genres[f"tag_vec_{i}"] = np.zeros(self.genres.shape[0])


        def getTagVec(row):
            return pd.Series(self.tag_vectoriser.queryDoc2VecModel(row['top_tags']))
        vecs = self.genres.apply(lambda row: getTagVec(row), axis = 1)
        self.genres[[f"tag_vec_{i}" for i in range(self.tv_len)]] = vecs
        self.save()

    def sperateGenres(self):
        """
        creates boolean coll for each genre
        """
        genres = "Action* Adventure* Animation* Children* Comedy* Crime* Documentary* Drama* Fantasy* Film-Noir* Horror* Musical* Mystery* Romance* Sci-Fi* Thriller* War* Western* (no genres listed)"
        genres = [g.replace(" ", "") for g in genres.split("*")]

        for g in genres:
            self.genres[g] = np.zeros(self.genres.shape[0])

        def assighnG(row, g):
            gs = row['genres'].split('|')
            if g in gs:
                return 1

        for g in genres:
            self.genres[g] = self.genres.apply(lambda row: assighnG(row, g), axis=1)


def main():
    plotParts = "data/movies_genres.csv"
    ml_tags_genome = "data/ml-25m/genome-scores.csv"
    ml_tags_lookup = "data/ml-25m/genome-tags.csv"
    ml_genres = "data/ml-25m/movies.csv"
    ml_ratings = "data/ml-25m/ratings.csv"

    tag_vec = load("data/temp/tag_vectorise")


    ml_tags = getCSV(ml_tags_lookup)

    load_stored_data = True
    if not load_stored_data:
        plotParts = getCSV(plotParts, sep="\t")
        ml_tags_genome = getCSV(ml_tags_genome)
        ml_genras = getCSV(ml_genres)



        items = ItemVec(plots=plotParts, tags=ml_tags_genome, tag_labels=ml_tags, genres=ml_genras, tag_vectoriser=tag_vec)
    else:
        items = ItemVec(None, None, ml_tags, None, load=True, tag_vectoriser=tag_vec)

    # ml_ratings = getCSV("data/ml-25m/ratings.csv")


if __name__ == '__main__':
    main()
