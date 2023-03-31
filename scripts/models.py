import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from vgg import VGGAutoEncoder, get_configs

from data import get_sample_patches_dataset


class DEC(nn.Module):
    """
    Deep Embedding Clustering model, uses the DEC_ConvAutoencoder as internal network


    Inputs:
        dims:
        n_clusters: int, number of clusters
        alpha: float, parameter of ...
        pretrain_epochs: int, number of epochs to pretrain the autoencoder
        train_epochs: int, number of epochs to train the model with the clustering loss
        plot_results: bool, if True, plot the latent space and the centers
        device: torch.device, device to use for training
    """

    def __init__(
        self,
        encoder,
        n_clusters=15,
        alpha=1,
        pretrain_epochs=100,
        train_epochs=100,
        plot_results=False,
        device=None,
    ):
        super().__init__()
        configs = get_configs(encoder)
        self.CAE = VGGAutoEncoder(configs=configs)
        self.embedding_size = 25088
        self.n_clusters = n_clusters
        self.alpha = alpha
        self.pretrain_epochs = pretrain_epochs
        self.train_epochs = train_epochs
        self.plot_results = plot_results
        self.device = device

        self.criterion = nn.MSELoss()
        self.cluster_criterion = nn.KLDivLoss(size_average=False)
        self.optimizer = torch.optim.Adam(self.CAE.parameters(), lr=1e-3)

    def get_centers(self, loader):
        """
        Obtain the centers of the clusters using KMeans as torch tensors.
        to do that, we obtain all the embeddings of the autoencoder, then apply KMeans (with cpu)

        Inputs:
            loader: torch.utils.data.DataLoader
        """
        auto_encoder_embeddings = []
        with torch.no_grad():
            for batch, _ in loader:
                batch = batch.to(self.device)
                embedding, _ = self.CAE(batch)
                auto_encoder_embeddings.append(embedding.cpu().detach().numpy())
        auto_encoder_embeddings = np.concatenate(auto_encoder_embeddings)

        k_means = KMeans(n_clusters=self.n_clusters, init="k-means++").fit(
            auto_encoder_embeddings
        )
        self.centers = nn.Parameter(
            torch.tensor(
                k_means.cluster_centers_, dtype=torch.float32, device=self.device
            ),
            requires_grad=True,
        )
        self.cluster_optimizer = torch.optim.SGD(
            params=list(self.CAE.parameters()) + [self.centers], lr=0.1, momentum=0.9
        )

        assert (self.centers[0, :] != self.centers[1, :]).sum() > 2

    def target_distribution(self, q_):
        """
        Obtain the distribution of the clusters based on the soft assignment.

        Inputs:
            q_: torch.tensor, shape (batch_size, n_clusters)
        """
        weight = (q_**2) / torch.sum(q_, 0)
        return (weight.t() / torch.sum(weight, 1)).t()

    def cluster_layer(self, embedding):
        """
        For a given embedding, compute the soft assignment of the clusters
        based on the distance to the centers.

        Inputs:
            embedding: torch.tensor, shape (batch_size, embedding_size)
        """
        norm_squared = torch.sum((embedding.unsqueeze(1) - self.centers) ** 2, 2)
        numerator = 1.0 / (1.0 + (norm_squared / self.alpha))
        power = float(self.alpha + 1) / 2
        numerator = numerator**power
        t_dist = numerator / torch.sum(
            numerator, dim=1, keepdim=True
        )  # soft assignment using t-distribution
        return t_dist

    def get_clusters_batch(self, batch):
        """
        Helper function to obtain the cluster assignment for a batch of data.
        It first computes the embeddings and then the soft assignment.

        Inputs:
            batch: torch.tensor, shape (batch_size, ...)
        """
        batch = batch.to(self.device)
        embedding, _ = self.CAE(batch)
        output = self.cluster_layer(embedding)
        return output.max(1)[1].cpu().detach().numpy()

    def get_clusters(self, loader):
        """
        Helper function to obtain the cluster assigment for a dataset.

        Inputs:
            loader: torch.utils.data.DataLoader
        """
        y = []
        with torch.no_grad():
            for batch, _ in loader:
                batch = batch.to(self.device)
                y.append(self.get_clusters_batch(batch))
        y = np.concatenate(y)
        return y

    def plot_latent_space(self, loader):
        """
        Plot the latent space using t-SNE, color based on the clusters.

        Inputs:
            loader: torch.utils.data.DataLoader
        """
        embeddings = []
        y = []
        with torch.no_grad():
            for batch, _ in loader:
                batch = batch.to(self.device)
                output = self.CAE(batch)[0]
                embeddings.append(output.cpu().detach().numpy())
                y.append(self.get_clusters_batch(batch))
        embeddings = np.concatenate(embeddings, axis=0)
        y = np.concatenate(y)
        embeddings_proj = TSNE(n_components=2).fit_transform(embeddings)

        plt.scatter(embeddings_proj[:, 0], embeddings_proj[:, 1], c=y)
        plt.show()

    def plot_centers(self):
        """
        Plot the centers as a line plot, each line is a center with the dimensions in the x axis.
        """
        for i in range(self.n_clusters):
            plt.plot(self.centers[i, :].cpu().detach().numpy(), label=f"center {i}")
        plt.legend()
        plt.show()

    def fit(self, loader):
        """
        Trainer function of the model, it pre-trains the autoencoder and then finish training with the clustering loss.

        Inputs:
            loader: torch.utils.data.DataLoader
        """

        # Pretrain Autoencoder
        print("Pretraining Autoencoder...")
        for _ in tqdm(range(self.pretrain_epochs)):
            for batch, _ in loader:
                batch = batch.to(self.device)
                _, reconstruction = self.CAE(batch)
                rec_loss = self.criterion(reconstruction, batch)
                rec_loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()

        # Train with clustering loss
        self.get_centers(loader)

        if self.plot_results:
            # Plot the pretrained latent space and the first centers
            self.eval()
            self.plot_latent_space(loader)
            self.plot_centers()
            self.train()

        print("Training with clustering loss...")
        for _ in tqdm(range(self.train_epochs)):
            for batch, _ in loader:
                batch = batch.to(self.device)
                embedding, _ = self.CAE(batch)
                output = self.cluster_layer(embedding)
                target = self.target_distribution(output).detach()
                loss = self.cluster_criterion(output.log(), target) / output.shape[0]
                self.cluster_optimizer.zero_grad()
                loss.backward()
                self.cluster_optimizer.step()

        if self.plot_results:
            # Plot the updated latent space and the updated centers
            self.eval()
            self.plot_latent_space(loader)
            self.plot_centers()




if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = get_sample_patches_dataset()
    print("dataset size:", len(dataset))
    dl = DataLoader(dataset, batch_size=16)
    model = DEC(
        "vgg16",
        n_clusters=15,
        alpha=1,
        pretrain_epochs=20,
        train_epochs=20,
        plot_results=True,
        device=device
    )
    model.to(device)
    model.fit(dl)

    centers = model.centers.cpu().detach().numpy()
    # save centers to npy
    np.save("centers.npy", centers)
