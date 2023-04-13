import os
import numpy as np
from tqdm import tqdm

from sklearn.cluster import KMeans
import torch
import torch.nn.functional as F
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader

from data import get_sample_patches_dataset
import utils as utils


class SmallAutoEncoder(nn.Module):
    def __init__(self, latent_dim = 64):
        super(SmallAutoEncoder, self).__init__()
        self.encoder_cnn = nn.Sequential(
            nn.Conv2d(3, 8, 3, stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv2d(8, 16, 3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.Conv2d(16, 32, 3, stride=2, padding=0),
            nn.ReLU(True)
        )
        
        ### Flatten layer
        self.flatten = nn.Flatten(start_dim=1)
        ### Linear section
        self.encoder_lin = nn.Sequential(
            nn.Linear(3 * 3 * 32, 128),
            nn.ReLU(True),
            nn.Linear(128, latent_dim)
        )

        self.decoder_lin = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(True),
            nn.Linear(128, 3 * 3 * 32),
            nn.ReLU(True)
        )

        self.unflatten = nn.Unflatten(dim=1, unflattened_size=(32, 3, 3))

        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose2d(32, 16, 3, stride=2, output_padding=0),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 8, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(True),
            nn.ConvTranspose2d(8, 3, 3, stride=2, padding=1, output_padding=1)
        )


    def forward(self, x):
        # encode 
        x = self.encoder_cnn(x)
        x = self.flatten(x)
        x = self.encoder_lin(x)
        encoded = x

        x = self.decoder_lin(x)
        x = self.unflatten(x)
        x = self.decoder_conv(x)
        decoded = torch.sigmoid(x)
        return encoded, decoded


class AutoEncoder(nn.Module):
    """
    AutoEncoder that uses a pretrained model as the encoder.

    Inputs:
        latent_dim: int with the dimension of the latent space !!! (must be a multiple of 49) !!!
        encoder_arch: string with the name of the pretrained model
        encoder_lock_weights: bool to lock the weights of the pretrained model
        decoder_latent_dim_channels: int with the number of channels of the latent space
        decoder_layers_per_block: list with the number of layers per block
        decoder_enable_bn: bool to enable batch normalization

    """

    def __init__(
        self,
        latent_dim,
        encoder_arch="vgg16",
        encoder_lock_weights=True,
        decoder_latent_dim_channels=128,
        decoder_layers_per_block=[2, 2, 3, 3, 3],
        decoder_enable_bn=False,
    ):
        super(AutoEncoder, self).__init__()
        #assert latent_dim % 49 == 0
        self.encoder = PetrainedEncoder(latent_dim, encoder_arch, encoder_lock_weights)
        self.decoder = Decoder(
            latent_dim,
            #decoder_latent_dim_channels,
            decoder_layers_per_block,
            decoder_enable_bn,
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded


class PetrainedEncoder(nn.Module):
    """
    Convolutional Encoder that uses a pretrained model as a base.
    It removes the last two layers and add two linear layers to generate the latent space.
    The pretrained model can be one of the following: vgg16, vgg19, resnet50, resnet152

    Inputs:
        latent_dim: int with the dimension of the latent space
        arch: string with the name of the pretrained model
        lock_weights: bool to lock the weights of the pretrained model
    """

    def __init__(self, latent_dim, arch="vgg16", lock_weights=True):
        super(PetrainedEncoder, self).__init__()
        assert arch in ["vgg16", "vgg19", "resnet50", "resnet152"]
        self.latent_dim = latent_dim
        #self.latent_channels = latent_dim // 49
        self.arch = arch
        self.lock_weights = lock_weights
        self.model = self._get_model()

    def _get_model(self):
        if self.arch == "vgg16":
            model = torchvision.models.vgg16(weights="DEFAULT")
        elif self.arch == "vgg19":
            model = torchvision.models.vgg19(weights="DEFAULT")
        elif self.arch == "resnet50":
            model = torchvision.models.resnet50(weights="DEFAULT")
        elif self.arch == "resnet152":
            model = torchvision.models.resnet152(weights="DEFAULT")

        if self.lock_weights:
            for param in model.parameters():
                param.requires_grad = False

        if "vgg" in self.arch:
            model = list(model.children())[:-1]
            model += [
                nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=0),
                nn.ReLU(),
                nn.Conv2d(256, 32, kernel_size=3, stride=1, padding=0),
                nn.ReLU(),
                nn.Flatten(1, -1),
                nn.Linear(32 * 3 * 3, 256),
                nn.ReLU(),
                nn.Linear(256, self.latent_dim),
            ]
        elif "resnet" in self.arch:
            model = list(model.children())[:-2]
            model += [
                nn.Conv2d(2048, 512, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=0),
                nn.ReLU(),
                nn.Conv2d(256, 32, kernel_size=3, stride=1, padding=0),
                nn.ReLU(),
                nn.Flatten(1, -1),
                nn.Linear(32 * 3 * 3, 256),
                nn.ReLU(),
                nn.Linear(256, self.latent_dim),
            ]

        model = nn.Sequential(*model)
        return model
    
    def forward(self, x):
        return self.model(x)


class Decoder(nn.Module):
    """
    Convolutional Decoder. It recieves an 1 dimensional vector, reshape into an image
    and apply convolutional layers to generate the image of size 224x224x3.

    Code adpted from https://github.com/Horizon2333/imagenet-autoencoder/blob/main/models/vgg.py

    Inputs:
        latent_dim: int with the dimension of the latent space !!! (must be a multiple of 49) !!!
        latent_dim_channels: int with the number of channels of the latent space (must be 128, 256 or 512)
        layers_per_block: list with the number of layers per block (must have 5 elements)
        enable_bn: bool to enable batch normalization
    """

    def __init__(
        self,
        latent_dim,
        #latent_dim_channels=128,
        layers_per_block=[2, 2, 3, 3, 3],
        enable_bn=False,
    ):
        super(Decoder, self).__init__()
        #assert latent_dim % 49 == 0
        assert len(layers_per_block) == 5
        #assert latent_dim_channels in [128, 256, 512]
        self.latent_dim = latent_dim
        #self.latent_channels = latent_dim // 49

        #self.latent_dim_channels = latent_dim_channels
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 7 * 7 * 8),
            nn.ReLU(),
        )
        self.conv1 = DecoderBlock(
                input_dim=8,
                output_dim=128,
                hidden_dim=64,
                layers=layers_per_block[0],
                enable_bn=enable_bn,
        )
        self.conv2 = DecoderBlock(
            input_dim=128,
            output_dim=256,
            hidden_dim=256,
            layers=layers_per_block[1],
            enable_bn=enable_bn,
        )
        self.conv3 = DecoderBlock(
            input_dim=256,
            output_dim=128,
            hidden_dim=128,
            layers=layers_per_block[2],
            enable_bn=enable_bn,
        )
        self.conv4 = DecoderBlock(
            input_dim=128,
            output_dim=64,
            hidden_dim=128,
            layers=layers_per_block[3],
            enable_bn=enable_bn,
        )
        self.conv5 = DecoderBlock(
            input_dim=64,
            output_dim=3,
            hidden_dim=64,
            layers=layers_per_block[4],
            enable_bn=enable_bn,
        )
        self.gate = nn.Sigmoid()

    def forward(self, x):
        x = self.fc(x)
        x = x.view(-1, 8, 7, 7)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.gate(x)
        return x


class DecoderBlock(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, layers, enable_bn=False):
        super(DecoderBlock, self).__init__()
        upsample = nn.ConvTranspose2d(
            in_channels=input_dim, out_channels=hidden_dim, kernel_size=2, stride=2
        )
        self.add_module("0 UpSampling", upsample)

        if layers == 1:
            layer = DecoderLayer(
                input_dim=input_dim, output_dim=output_dim, enable_bn=enable_bn
            )
            self.add_module("1 DecoderLayer", layer)
        else:
            for i in range(layers):
                if i == 0:
                    layer = DecoderLayer(
                        input_dim=input_dim, output_dim=hidden_dim, enable_bn=enable_bn
                    )
                elif i == (layers - 1):
                    layer = DecoderLayer(
                        input_dim=hidden_dim, output_dim=output_dim, enable_bn=enable_bn
                    )
                else:
                    layer = DecoderLayer(
                        input_dim=hidden_dim, output_dim=hidden_dim, enable_bn=enable_bn
                    )
                self.add_module("%d DecoderLayer" % (i + 1), layer)

    def forward(self, x):
        for name, layer in self.named_children():
            x = layer(x)
        return x


class DecoderLayer(nn.Module):
    def __init__(self, input_dim, output_dim, enable_bn):
        super(DecoderLayer, self).__init__()
        if enable_bn:
            self.layer = nn.Sequential(
                nn.BatchNorm2d(input_dim),
                nn.ReLU(inplace=True),
                nn.Conv2d(
                    in_channels=input_dim,
                    out_channels=output_dim,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                ),
            )
        else:
            self.layer = nn.Sequential(
                nn.ReLU(inplace=True),
                nn.Conv2d(
                    in_channels=input_dim,
                    out_channels=output_dim,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                ),
            )

    def forward(self, x):
        return self.layer(x)


class DEC(nn.Module):
    """
    Deep Embedding Clustering model, uses the DEC_ConvAutoencoder as internal network


    Inputs:
        latent_dim: int with the dimension of the latent space
        encoder_arch: string with the name of the pretrained model
        encoder_lock_weights: bool to lock the weights of the pretrained model
        decoder_latent_dim_channels: int with the number of channels of the latent space
        decoder_layers_per_block: list with the number of layers per block
        n_clusters: int, number of clusters
        alpha: float, parameter of ...
        pretrain_epochs: int, number of epochs to pretrain the autoencoder
        train_epochs: int, number of epochs to train the model with the clustering loss
        plot_results: bool, if True, plot the latent space and the centers
        results_dir: string, directory to save the results
        device: torch.device, device to use for training
    """

    def __init__(
        self,
        latent_dim,
        encoder_arch="vgg16",
        encoder_lock_weights=True,
        decoder_latent_dim_channels=128,
        decoder_layers_per_block=[2, 2, 3, 3, 3],
        n_clusters=15,
        alpha=1,
        pretrain_epochs=100,
        train_epochs=100,
        plot_results=False,
        results_dir=None,
        device=None,
    ):
        super().__init__()
        self.CAE = AutoEncoder(
            latent_dim,
            encoder_arch,
            encoder_lock_weights,
            decoder_latent_dim_channels,
            decoder_layers_per_block,
        )
        self.embedding_size = latent_dim
        self.n_clusters = n_clusters
        self.alpha = alpha
        self.pretrain_epochs = pretrain_epochs
        self.train_epochs = train_epochs
        self.plot_results = plot_results
        self.device = device

        self.criterion = nn.MSELoss()
        self.cluster_criterion = nn.KLDivLoss(size_average=False)
        self.optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.CAE.parameters()), lr=1e-3
        )

        # if exists results dir
        if results_dir is not None:
            if not os.path.exists(f"../models/{results_dir}"):
                os.makedirs(f"../models/{results_dir}")
            self.results_dir = f"../models/{results_dir}"

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

    def get_embeddings_labels(self, loader):
        """
        Helper function to obtain the embeddings and the labels for a dataset.

        Inputs:
            loader: torch.utils.data.DataLoader
        """
        embeddings = []
        labels = []
        filenames = []
        with torch.no_grad():
            for batch, filename in loader:
                batch = batch.to(self.device)
                embedding, _ = self.CAE(batch)
                embeddings.append(embedding.cpu().detach().numpy())
                filenames.append(filename)
                labels.append(self.get_clusters_batch(batch))
        embeddings = np.concatenate(embeddings, axis=0)
        labels = np.concatenate(labels)
        filenames = np.concatenate(filenames)
        return embeddings, labels, filenames

    def fit(self, loader):
        """
        Trainer function of the model, it pre-trains the autoencoder and then finish training with the clustering loss.

        Inputs:
            loader: torch.utils.data.DataLoader
        """
        # Pretrain Autoencoder
        print("Pretraining Autoencoder...")
        pretraining_losses = []
        for _ in tqdm(range(self.pretrain_epochs)):
            iter_loss = 0
            for batch, _ in loader:
                batch = batch.to(self.device)
                _, reconstruction = self.CAE(batch)
                rec_loss = self.criterion(reconstruction, batch)
                rec_loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
                iter_loss += rec_loss.item()
            pretraining_losses.append(iter_loss)

        # Train with clustering loss
        self.get_centers(loader)

        if self.plot_results:
            self.eval()
            embeddings, labels, filenames = self.get_embeddings_labels(loader)
            utils.embedding_proj(
                embeddings,
                labels,
                os.path.join(self.results_dir, f"pretrain_latent_space.png"),
            )
            utils.loss_curve(
                pretraining_losses, os.path.join(self.results_dir, f"pretrain_loss.png")
            )
            np.save(os.path.join(self.results_dir, "pretrain_labels.npy"), labels)
            np.save(os.path.join(self.results_dir, "pretrain_filenames.npy"), filenames)
            self.train()

        print("Training with clustering loss...")
        train_losses = []
        for _ in tqdm(range(self.train_epochs)):
            iter_loss = 0
            for batch, _ in loader:
                batch = batch.to(self.device)
                embedding, _ = self.CAE(batch)
                output = self.cluster_layer(embedding)
                target = self.target_distribution(output).detach()
                loss = self.cluster_criterion(output.log(), target) / output.shape[0]
                self.cluster_optimizer.zero_grad()
                loss.backward()
                self.cluster_optimizer.step()
                iter_loss += loss.item()
            train_losses.append(iter_loss)

        if self.plot_results:
            # Plot the updated latent space and the updated centers
            self.eval()
            embeddings, labels, filenames = self.get_embeddings_labels(loader)
            utils.embedding_proj(
                embeddings,
                labels,
                os.path.join(self.results_dir, f"latent_space.png"),
            )
            utils.loss_curve(
                train_losses, os.path.join(self.results_dir, f"train_loss.png")
            )
            np.save(os.path.join(self.results_dir, "train_labels.npy"), labels)
            np.save(os.path.join(self.results_dir, "train_filenames.npy"), filenames)
            # save model


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    filenames = os.listdir("../data/output/patches")
    filenames = [os.path.join("../data/output/patches", f) for f in filenames]
    dataset = get_sample_patches_dataset(filenames=filenames)
    print("Dataset Size:", len(dataset))
    dl = DataLoader(dataset, batch_size=64, shuffle=True)
    model = DEC(
        latent_dim=128,
        encoder_arch="resnet50",
        n_clusters=5,
        alpha=1,
        pretrain_epochs=10,
        train_epochs=10,
        plot_results=True,
        results_dir="DEC_resnet50_results",
        device=device,
    )

    encoder = model.CAE.encoder
    pytorch_trainable_params = sum(
        p.numel() for p in encoder.parameters() if p.requires_grad
    )
    print("Millions of encoder trainable parameters:", pytorch_trainable_params // 1e6)
    decoder = model.CAE.decoder
    pytorch_trainable_params = sum(
        p.numel() for p in decoder.parameters() if p.requires_grad
    )
    print("Millions of decoder trainable parameters:", pytorch_trainable_params // 1e6)

    model.to(device)
    model.fit(dl)

    # Save model
    torch.save(model.state_dict(), "../models/DEC_resnet50_results/DEC_resnet50.pth")
