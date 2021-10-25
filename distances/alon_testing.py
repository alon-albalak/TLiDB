from utils import datasets
from dataset_distance import gather_utterances
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import numpy as np

from scipy.stats import entropy
from scipy.spatial.distance import jensenshannon
from scipy.special import kl_div

def vectorize_print(vectorizer, corpus_to_fit,texts_to_transform):
    vectorizer.fit(corpus_to_fit)
    X = vectorizer.transform(texts_to_transform)
    # X = vectorizer.fit_transform(combined)
    # print(X.data[0])
    # print(X[0])
    # print(X[0,1133])

    X = np.array(X.toarray(), dtype=float)
    # print(X[0])
    # print(X[0,1133])

    # print(utts[0])
    # print(utts[1])
    print(cos_sim(X[0], X[0]))
    print(cos_sim(X[0], X[1]))
    print(cos_sim(X[0], X[2]))
    print(cos_sim(X[1], X[2]))
    print()
    eps = 1e-12
    X += eps
    print(entropy(X[0]))
    print(entropy(X[0], X[1]))
    print(entropy(X[0], X[2]))
    print(entropy(X[1], X[2]))
    print()
    # print(jensenshannon(X[0]))
    print(jensenshannon(X[0], X[1]))
    print(jensenshannon(X[0], X[2]))
    print(jensenshannon(X[1], X[2]))
    print()
    print(np.sum(kl_div(X[0], X[1])))
    print(np.sum(kl_div(X[0], X[2])))
    print(np.sum(kl_div(X[1], X[2])))


def cos_sim(a,b):
    return np.dot(a,b) / (np.linalg.norm(a)*np.linalg.norm(b))

dataset = datasets.load_dataset("multiwoz22")
utts = gather_utterances(dataset['train_dialogues_001'])
utts2 = gather_utterances(dataset['train_dialogues_002'])
utts3 = gather_utterances(dataset['test_dialogues_001'])
corpus_to_fit = utts+utts2+utts3
texts_to_transform = [" ".join(utts), " ".join(utts2), " ".join(utts3)]
print("TfIdf vectorizer")
vectorizer = TfidfVectorizer(stop_words="english",lowercase=True, ngram_range=(1,1),smooth_idf=True)
vectorize_print(vectorizer,corpus_to_fit,texts_to_transform)


print("Count vectorizer")
vectorizer = CountVectorizer(stop_words="english", lowercase=True, ngram_range=(1, 1))
vectorize_print(vectorizer, corpus_to_fit, texts_to_transform)
