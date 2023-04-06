import numpy as np
import matplotlib.pyplot as plt

from sklearn.manifold import TSNE


def embedding_proj(embeddings, labels=None, dir=None):
    """
    Plot the latent space using t-SNE, color based on the clusters.

    Inputs:
        embeddings - np.array of shape (n_samples, n_features)
        labels - np.array of shape (n_samples, )
        dir - str, directory to save the plot
    """

    embeddings_proj = TSNE(n_components=2).fit_transform(embeddings)

    if labels is None:
        labels = np.zeros(embeddings.shape[0])
    plt.scatter(embeddings_proj[:, 0], embeddings_proj[:, 1], c=labels, cmap="tab10")
    if dir is not None:
        plt.savefig(dir)
        plt.close()
    else:
        plt.show()

def loss_curve(loss, dir=None):
    """
    Plot the loss curve.

    Inputs:
        loss - list or numpy array (n_epochs, )
        dir - str, directory to save the plot
    """
    plt.plot(loss)
    plt.ylabel("Iter loss")
    plt.xlabel("Epoch")
    plt.title("Training loss")
    if dir is not None:
        plt.savefig(dir)
        plt.close()
    else:
        plt.show()
