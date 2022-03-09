from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

def plot_tsne(embedings, words):
    X = TSNE(perplexity=5.0).fit_transform(embedings)
    plt.scatter(X[:, 0], X[:, 1])
    for w, (x, y) in zip(words, X):
        plt.text(x, y, "\n".join(w))
    plt.show()