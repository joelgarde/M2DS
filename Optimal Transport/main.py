from itertools import chain

import numpy as np

from src.consts import REUTERS_P, GLOVE_P
from src.dataset import DataLoader
from src.encode_reuters import fit_glove, make_train_test, load_glove
from src.jax_model import init_train, do_train, load_topics
from src.parse_reuters import reuters
from src.pull import download, download_glove, REUTERS_URL
from src.tsne import plot_tsne


def setup_reuters_data():
    download(REUTERS_P, REUTERS_URL)
    download_glove(GLOVE_P)
    data = fit_glove()

if __name__ == '__main__':

    train = False
    visualize = True

    if train:
        X_train, X_test = make_train_test()
        print(f"training on {X_train.shape[0]} exemples.")
        loader = DataLoader(X_train)
        train_step, train_state = init_train()
        do_train(train_step, train_state, loader)

    if visualize:
        word_embeddings, topic_embeddings = load_topics()
        voc = load_glove()
        assert len(voc.voc) == word_embeddings.shape[0]
        print(topic_embeddings.shape, word_embeddings.shape)
        print("analysing top words per topic.")
        similarity = topic_embeddings @ word_embeddings.T
        n_words = 4
        best_words_idx = np.argpartition(similarity, kth=-n_words)[:, -n_words:]
        best_words = voc.voc[best_words_idx]

        for topic in best_words:
            print(" ".join(topic))

        plot_tsne(topic_embeddings, best_words)