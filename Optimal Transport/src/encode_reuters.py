import dataclasses
from collections import defaultdict
from dataclasses import dataclass
from itertools import islice
from pathlib import Path
import pickle
from typing import Tuple, Union, Dict

import numpy as np
import scipy.sparse
from sklearn.feature_extraction.text import HashingVectorizer, TfidfVectorizer

from src.consts import TFIDF_P, VOC_P, GLOVE_REUTERS_P, TRAIN_TEST_P
from src.glove import glove_vocab, glove_vector
from src.parse_reuters import reuters


@dataclass
class VectorizedText:
    X: scipy.sparse.csr_matrix
    voc: Union[dict, defaultdict]


@dataclass
class EmbeddedText:
    X: scipy.sparse.csr_matrix
    voc: np.array
    E: np.array

    def __post_init__(self):
        assert self.X.shape[1] == len(self.voc) == self.E.shape[0]


def tfidf() -> VectorizedText:
    if not TFIDF_P.exists():
        glove_voc = {w: i for i, w in enumerate(glove_vocab())}
        encoder = TfidfVectorizer(strip_accents="unicode", stop_words="english", norm="l1", # l1 TO NORMALIZE AS PROBABILITIES
                                  use_idf=False, min_df=0.01, max_df=0.9)
        text = reuters().text
        tfidf = encoder.fit_transform(text, None)
        scipy.sparse.save_npz(TFIDF_P, tfidf)
        pickle.dump(encoder.vocabulary_, open(VOC_P, "wb"))
        return VectorizedText(tfidf, encoder.vocabulary_)
    else:
        return VectorizedText(scipy.sparse.load_npz(TFIDF_P), pickle.load(open(VOC_P, "rb")))


def fit_glove() -> EmbeddedText:
    if not GLOVE_REUTERS_P.exists():
        X, voc = dataclasses.astuple(tfidf())
        glove = dict(zip(glove_vocab(), glove_vector()))
        vec = np.vstack(list((glove[w] for w in voc if w in glove)))
        in_glove = {w: i for w, i in voc.items() if w in glove}
        known = np.array(list(in_glove.values()))
        filtered_voc = np.array(list(in_glove.keys()))
        print(f"{len(voc) - len(filtered_voc)}")
        X_filtered = X[:, known] # filter out words that are not in glove. There is only 2 ( reuters specific abreviations).
        pickle.dump(EmbeddedText(X_filtered, filtered_voc, vec), open(GLOVE_REUTERS_P, "wb"))
    return pickle.load(open(GLOVE_REUTERS_P, "rb"))

def load_glove() -> EmbeddedText:
    with open(GLOVE_REUTERS_P, "rb") as f:
        d = pickle.load(f)
    return d

def reuters_embeddings() -> np.array:
    return fit_glove().E


def make_train_test():
    if not TRAIN_TEST_P.exists():
        embedded = fit_glove()
        original_data = reuters()
        split = np.asarray(original_data.splits)
        empty = (embedded.X.getnnz(1) == 0)
        print(empty.sum())
        train, test = embedded.X[(split == "TRAIN") & (~empty)], embedded.X[(split == "TEST") & (~empty)]
        with open(TRAIN_TEST_P, "wb") as f:
            pickle.dump((train, test), f)
    with open(TRAIN_TEST_P, "rb") as f:
        train, test = pickle.load(f)
    return train, test