import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans


class ConvAutoencoder(nn.Module):
    """
    Convolutional Autoencoder using the VGG16 architecture as enconder. (to use few parameters)
    """

    def __init__(self, dims):
        super(ConvAutoencoder, self).__init__()

        # Encoder
        self.encoder_layer1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.encoder_layer2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.encoder_layer3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )
        self.encoder_layer4 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.encoder_layer5 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )
        self.encoder_layer6 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )
        self.encoder_layer7 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.encoder_layer8 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
        )
        self.encoder_layer9 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
        )
        self.encoder_layer10 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.encoder_layer11 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
        )
        self.encoder_layer12 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
        )
        self.encoder_layer13 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.encoder_fc1 = nn.Sequential(
            nn.Dropout(0.5), nn.Linear(7 * 7 * 512, 4096), nn.ReLU()
        )
        self.encoder_fc2 = nn.Sequential(
            nn.Dropout(0.5), nn.Linear(4096, 4096), nn.ReLU()
        )

        # Decoder

        # self.decoder = []
        # for i, (input, output) in enumerate(zip(dims[:-1], dims[1:])):

        #    self.t_conv1 = nn.ConvTranspose2d(4, 16, 2, stride=2)

        # dims_inverse = dims[::-1]
        # for (input, output) in zip(dims_inverse[:-1], dims_inverse[1:]):
        #    self.decoder.append(nn.Linear(input, output))
        #    self.t_conv2 = nn.ConvTranspose2d(16, 3, 2, stride=2)
        #    if i != len(dims) - 2:
        #        self.decoder.append(nn.ReLU())
        #
        # self.decoder = nn.Sequential(*self.decoder)

    def forward(self, x):
        x = self.encoder_layer1(x)
        x = self.encoder_layer2(x)
        x = self.encoder_layer3(x)
        x = self.encoder_layer4(x)
        x = self.encoder_layer5(x)
        x = self.encoder_layer6(x)
        x = self.encoder_layer7(x)
        x = self.encoder_layer8(x)
        x = self.encoder_layer9(x)
        x = self.encoder_layer10(x)
        x = self.encoder_layer11(x)
        x = self.encoder_layer12(x)
        x = self.encoder_layer13(x)
        x = x.view(x.size(0), -1)
        x = self.encoder_fc1(x)
        encoded = self.encoder_fc2(x)

        return encoded


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
        dims,
        n_clusters=15,
        alpha=1,
        pretrain_epochs=100,
        train_epochs=100,
        plot_results=False,
        device=None,
    ):
        super().__init__()
        self.CAE = ConvAutoencoder(dims)
        self.embedding_size = dims[-1]
        self.n_clusters = n_clusters
        self.alpha = alpha
        self.pretrain_epochs = pretrain_epochs
        self.train_epochs = train_epochs
        self.plot_results = plot_results
        self.device = device

        self.criterion = nn.MSELoss()
        self.cluster_criterion = nn.KLDivLoss(size_average=False)
        self.optimizer = torch.optim.Adam(self.AE.parameters(), lr=1e-3)

    def get_centers(self, loader):
        """
        Obtain the centers of the clusters using KMeans as torch tensors.
        to do that, we obtain all the embeddings of the autoencoder, then apply KMeans (with cpu)

        Inputs:
            loader: torch.utils.data.DataLoader
        """
        auto_encoder_embeddings = []
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
            params=list(self.AE.parameters()) + [self.centers], lr=0.1, momentum=0.9
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
        for batch, _ in loader:
            batch = batch.to(self.device)
            output = self.AE(batch)[0]
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
        for _ in range(self.pretrain_epochs):
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

        for _ in range(self.train_epochs):
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
    print("hi")
    sample_img = torch.rand(224, 224, 3)
    cae = ConvAutoencoder()

    out = cae(sample_img.unsqueeze(0))
    print(out.shape)
