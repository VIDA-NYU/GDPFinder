import os
import numpy as np
import torch
from torchvision import transforms
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans


def get_embeddings(loader, model, device):
    """
    Helper function to obtain the embeddings for a dataset.

    Inputs:
        loader: torch.utils.data.DataLoader
    """
    embeddings = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            embedding = model.encoder(batch)
            embeddings.append(embedding.cpu().detach().numpy())
    embeddings = np.concatenate(embeddings, axis=0)
    return embeddings


def get_clusters(loader, model, device):
    clusters = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            cluster = model(batch)
            clusters.append(cluster.cpu().detach().numpy())
    clusters = np.concatenate(clusters, axis=0)
    clusters = clusters.argmax(axis=1)
    return clusters


def cluster_embeddings(embeddings, n_clusters):
    cl = KMeans(n_clusters=n_clusters, n_init=20).fit(embeddings)
    return cl.cluster_centers_


def plot_embedding_proj(embeddings, labels=None, dir=None):
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


def plot_loss_curve(losses_log, dir=None):
    """
    Plot the loss curve.

    Inputs:
        loss - list or numpy array (n_epochs, )
        dir - str, directory to save the plot
    """
    plt.plot(losses_log)
    plt.ylabel("Iter loss")
    plt.xlabel("Epoch")
    plt.title("Training loss")
    if dir is not None:
        plt.savefig(dir)
        plt.close()
    else:
        plt.show()


def plot_reconstruction(model, dl, device, n_samples=5, dir=None):
    imgs = []
    reconstructions = []
    k = 0
    for batch in dl:
        for j in range(batch.shape[0]):
            img = batch[j].unsqueeze(0)
            imgs.append(img.cpu().detach().numpy())
            img = img.to(device)
            reconstruction = model(img)
            reconstructions.append(reconstruction.cpu().detach().numpy())
            k += 1
            if k == n_samples:
                break
        if k == n_samples:
            break

    imgs = np.concatenate(imgs, axis=0)
    reconstructions = np.concatenate(reconstructions, axis=0)

    fig, axs = plt.subplots(nrows=n_samples, ncols=2, figsize=(4, 15))
    for i in range(n_samples):
        img_ = imgs[i].transpose(1, 2, 0)
        img_ = np.clip(img_, 0, 1)
        rec_ = reconstructions[i].transpose(1, 2, 0)
        rec_ = np.clip(rec_, 0, 1)

        axs[i, 0].imshow(img_)
        axs[i, 1].imshow(rec_)
    if dir is not None:
        plt.savefig(dir)
        plt.close()
    else:
        plt.show()


def save_reconstruction_results(
    mode, losses_log, batches_log, dl, model, device, dir=None
):
    # verify if the directory exists
    if dir is not None and not os.path.exists(dir):
        os.mkdir(dir)

    plot_loss_curve(losses_log, dir=dir + "loss_curve.png")
    plot_loss_curve(batches_log, dir=dir + "batch_loss_curve.png")
    torch.save(model.state_dict(), dir + "model.pt")
    if mode == "reconstruction":
        plot_reconstruction(model, dl, device, dir=dir + "reconstruction.png")
 
