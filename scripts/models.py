import os
import numpy as np
from tqdm import tqdm

from sklearn.cluster import KMeans
import torch
import torch.nn.functional as F
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader


class SmallAutoEncoder(nn.Module):
    def __init__(self, latent_dim=64):
        super(SmallAutoEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 8, 3, stride=2, padding=1),
            nn.ReLU(True),
            #nn.Conv2d(8, 8, 3, stride=1, padding=1),
            #nn.ReLU(True),
            #nn.Conv2d(8, 8, 3, stride=1, padding=1),
            #nn.ReLU(True),
            #nn.Conv2d(8, 8, 3, stride=1, padding=1),
            #nn.ReLU(True),
            nn.Conv2d(8, 16, 3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.Conv2d(16, 32, 3, stride=2, padding=0),
            nn.ReLU(True),
            nn.Flatten(1),
            nn.Linear(3 * 3 * 32, 128),
            nn.ReLU(True),
            nn.Linear(128, latent_dim),
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(True),
            nn.Linear(128, 3 * 3 * 32),
            nn.ReLU(True),
            nn.Unflatten(dim=1, unflattened_size=(32, 3, 3)),
            nn.ConvTranspose2d(32, 16, 3, stride=2, output_padding=0),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 8, 3, stride=2, padding=1, output_padding=1),
            #nn.Conv2d(8, 8, 3, stride=1, padding=1),
            #nn.ReLU(True),
            #nn.Conv2d(8, 8, 3, stride=1, padding=1),
            #nn.ReLU(True),
            nn.BatchNorm2d(8),
            nn.ReLU(True),
            nn.ConvTranspose2d(8, 3, 3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        # encode
        x = self.encoder(x)
        encoded = x

        # decode
        decoded = self.decoder(x)
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
        # assert latent_dim % 49 == 0
        self.encoder = PetrainedEncoder(latent_dim, encoder_arch, encoder_lock_weights)
        self.decoder = Decoder(
            latent_dim,
            # decoder_latent_dim_channels,
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
        # self.latent_channels = latent_dim // 49
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
        # latent_dim_channels=128,
        layers_per_block=[2, 2, 3, 3, 3],
        enable_bn=False,
    ):
        super(Decoder, self).__init__()
        # assert latent_dim % 49 == 0
        assert len(layers_per_block) == 5
        # assert latent_dim_channels in [128, 256, 512]
        self.latent_dim = latent_dim
        # self.latent_channels = latent_dim // 49

        # self.latent_dim_channels = latent_dim_channels
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
    def __init__(
        self, n_clusters, embedding_dim, encoder, cluster_centers=None, alpha=1.0
    ):
        """
        Module which holds all the moving parts of the DEC algorithm, as described in
        Xie/Girshick/Farhadi; this includes the AutoEncoder stage and the ClusterAssignment stage.

        Inputs:
            n_clusters: int, number of clusters
            embedding_dim, encoder part of the AutoEncoder
            cluster_centers: torch.tensor, shape (n_clusters, embedding_dim), initial cluster centers
            alpha: float, parameter of the t-distribution
        """
        super(DEC, self).__init__()
        self.encoder = encoder
        self.embedding_dim = embedding_dim
        self.cluster_number = n_clusters
        self.alpha = alpha
        self.assignment = ClusterAssignment(
            n_clusters, embedding_dim, alpha, cluster_centers
        )

    def forward(self, batch):
        """
        Compute the cluster assignment using the ClusterAssignment after running the batch
        through the encoder part of the associated AutoEncoder module.

        Inputs:
            batch: torch.tensor, shape (batch_size, embedding_dim)

        Outputs:
            torch.tensor, shape (batch_size, n_clusters)
        """
        return self.assignment(self.encoder(batch))


class ClusterAssignment(nn.Module):
    def __init__(
        self,
        n_clusters,
        embedding_dim,
        alpha=1,
        cluster_centers=None,
    ):
        """
        Module to handle the soft assignment, for a description see in 3.1.1. in Xie/Girshick/Farhadi,
        where the Student's t-distribution is used measure similarity between feature vector and each
        cluster centroid.

        Inputs:
            n_clusters - int, number of clusters
            embedding_dimension - int, dimension of the embedding
            alpha - float, parameter of the t-distribution
            cluster_centers - torch.tensor, shape (n_clusters, embedding_size)

        """
        super(ClusterAssignment, self).__init__()
        self.embedding_dim = embedding_dim
        self.n_clusters = n_clusters
        self.alpha = alpha
        if cluster_centers is None:
            initial_cluster_centers = torch.zeros(
                self.n_clusters, self.embedding_dim, dtype=torch.float
            )
            nn.init.xavier_uniform_(initial_cluster_centers)
        else:
            initial_cluster_centers = cluster_centers
        self.cluster_centers = nn.Parameter(initial_cluster_centers)

    def forward(self, batch):
        """
        Compute the soft assignment for a batch of feature vectors, returning a batch of assignments
        for each cluster.

        Inputs:
            batch - torch.tensor, shape (batch_size, embedding_size)

        Outputs:
            torch.tensor, shape (batch_size, n_clusters)
        """
        norm_squared = torch.sum((batch.unsqueeze(1) - self.cluster_centers) ** 2, 2)
        numerator = 1.0 / (1.0 + (norm_squared / self.alpha))
        power = float(self.alpha + 1) / 2
        numerator = numerator**power
        return numerator / torch.sum(numerator, dim=1, keepdim=True)


def target_distribution(batch):
    """
    Compute the target distribution p_ij, given the batch (q_ij), as in 3.1.3 Equation 3 of
    Xie/Girshick/Farhadi; this is used the KL-divergence loss function.

    Inputs:
        batch - [batch size, number of clusters] tensor of cluster assigments

    Outputs:
        [batch size, number of clusters] tensor of target distribution
    """
    weight = (batch**2) / torch.sum(batch, 0)
    return (weight.t() / torch.sum(weight, 1)).t()
