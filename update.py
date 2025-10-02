import numpy as np
import nltk

nltk.download("punkt")
import regex
import sys
from pyvi import ViTokenizer
from gensim.models import KeyedVectors
from gensim.models import Word2Vec
from gensim.test.utils import lee_corpus_list
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt


# import sklearn.feature_extraction.text.TfidfTransformer
def getData(filePath):
    with open(filePath, encoding="utf-8") as f:
        contents = f.read()
    return contents


def getStop_words(filePath_stop_word):
    with open(filePath_stop_word, encoding="utf-8") as f:
        stop_words = f.read()
        # print (stop_words)
    stop_words = stop_words.split("\n")
    # print(stop_words)
    return stop_words


def preProcessing(contents):
    # chuyen chu hoa thanh chu thuong
    contents = contents.lower()
    # loai bo nhung ki tu thua
    contents = contents.replace("\n", " ")
    # loai bo di khoang trang thua`
    contents = contents.strip()
    # loai bo cac emoji
    # contents = regex.compile(
    #     "["
    #     "\U0001F600-\U0001F64F"  # emoticons
    #     "\U0001F300-\U0001F5FF"  # symbols & pictographs
    #     "\U0001F680-\U0001F6FF"  # transport & map symbols
    #     "\U0001F1E0-\U0001F1FF"  # flags (iOS)
    #     "\U00002500-\U00002BEF"  # chinese char
    #     "\U00002702-\U000027B0"
    #     "\U00002702-\U000027B0"
    #     "\U000024C2-\U0001F251"
    #     "\U0001f926-\U0001f937"
    #     "\U00010000-\U0010ffff"
    #     "\u2640-\u2642"
    #     "\u2600-\u2B55"
    #     "\u200d"
    #     "\u23cf"
    #     "\u23e9"
    #     "\u231a"
    #     "\ufe0f"  # dingbats
    #     "\u3030"
    #     "]+",
    #     flags=regex.UNICODE,
    # )

    # print(contents)
    return contents


# ham xu ly tach cau trong van ban
def division(contents):
    sentences = nltk.sent_tokenize(contents)
    # print(sentences)
    return sentences


# ham lay tong de duoc vector cho tung cau
def sentenceVector(sentences, stop_words):
    # tao vector cho tu bang mo hinh word2vec
    w2v = KeyedVectors.load_word2vec_format("vi_txt/vi.vec", binary=False)
    # lay danh sach casc tuw trong tu dien
    # model = Word2Vec(w2v, vector_size=24, epochs=100)
    vocab = w2v.key_to_index
    # vocab = w2v.wv
    # khoi tao list luu tru vector dai dien cho tung cau
    X = []
    # duyet tung cau trong daon van
    for sentence in sentences:
        # su dung Vitokenizer de tach tu ghep trong tieng viet
        sentence_tokenizer = ViTokenizer.tokenize(sentence)
        # print (sentence_tokenizer)
        words = sentence_tokenizer.split(" ")
        # khoi tao vector 100 chieu
        sentence_vec = np.zeros((100))
        num_word = 0
        for word in words:
            # chi vector hoa khi tu khong nam trong  stop_words
            if word in vocab and word not in stop_words:
                num_word += 1
                # thiet lap vector dai dien cho ca cau bang cach lay tong cac vector dai dien cho tu trong cau
                sentence_vec += w2v.key_to_index[word]
        # print(num_word)
        X.append(sentence_vec / num_word)
    return X


def sentecesCluster(X):
    # lay so cau can tom tat bang 30% tong so cau
    n_clusters = len(X) * 50 // 100
    # phan cum cac caau
    kmeans = KMeans(n_clusters=n_clusters)
    kmeans = kmeans.fit(X)
    # print(kmeans)
    return kmeans


# xay dung doan van tom tat
def buildSummary(kmeans, X, sentences):
    n_clusters = len(X) * 30 // 100
    avg = []
    for j in range(n_clusters):
        idx = np.where(kmeans.labels_ == j)[0]
        avg.append(np.mean(idx))
    # print(avg)
    closest, _ = pairwise_distances_argmin_min(kmeans.cluster_centers_, X)
    ordering = sorted(range(n_clusters), key=lambda k: avg[k])
    summary = " ".join([sentences[closest[idx]] for idx in ordering])
    return summary


def summarizations(contents):
    filePath_stop_word = "stopword.txt"
    stop_words = getStop_words(filePath_stop_word)

    # contents = ''
    # contents = getData(filePath)
    # print (contents)
    contents = preProcessing(contents)
    sentences = division(contents)
    X = sentenceVector(sentences, stop_words)
    # print(np.shape(X))
    # print(X[0])
    kmeans = sentecesCluster(X)
    summary = buildSummary(kmeans, X, sentences)
    print("\n")

    # getNumSentences(X
    # print(summary)
    return summary
