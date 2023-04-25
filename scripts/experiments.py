import os
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader

from data import get_sample_patches_dataset, get_filenames
from models import AutoEncoder, SmallAutoEncoder, DEC
from train import train_reconstruction, train_clustering
from utils import save_reconstruction_results, get_embeddings, cluster_embeddings


def reconstruction_experiment():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    filenames = get_filenames()
    dataset = get_sample_patches_dataset(filenames=filenames, resize=(28, 28))
    dl = DataLoader(dataset, batch_size=256, shuffle=True)

    print("Training AutoEncoder ...")
    print("===================================")
    print(f"Dataset shape: {len(dataset)}")

    # model = AutoEncoder(
    #    latent_dim=128,
    #    encoder_arch="vgg16",
    #    encoder_lock_weights=True,
    #    decoder_latent_dim_channels=128,
    #    decoder_layers_per_block=[2, 2, 2, 2, 2],
    #    decoder_enable_bn=False,
    # ).to(device)
    model = SmallAutoEncoder(64).to(device)

    print(
        f"Number of parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)//1000000:d}M"
    )

    loss = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    losses_log, batches_log = train_reconstruction(
        model, dl, loss, optimizer, device, epochs=20
    )
    save_reconstruction_results(
        "reconstruction",
        losses_log,
        batches_log,
        dl,
        model,
        device,
        dir="../models/AE_small/",
    )


def clustering_experiment():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    filenames = get_filenames()
    dataset = get_sample_patches_dataset(filenames=filenames, resize=(28, 28))
    dl = DataLoader(dataset, batch_size=256)  # , shuffle=True)

    print("Training AutoEncoder ...")
    print("===================================")
    print(f"Dataset shape: {len(dataset)}")

    model_autoencoder = SmallAutoEncoder(64).to(device)
    model_autoencoder.load_state_dict(torch.load("../models/AE_small/model.pt"))
    model_autoencoder.eval()
    embeddings = get_embeddings(dl, model_autoencoder, device)
    centers = torch.tensor(cluster_embeddings(embeddings, 10))
    encoder = model_autoencoder.encoder

    model = DEC(
        n_clusters=10, embedding_dim=64, encoder=encoder, cluster_centers=centers
    )
    model.to(device)

    print(
        f"Number of parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)//1000000:d}M"
    )

    loss = nn.KLDivLoss(size_average=False)

    optimizer = torch.optim.SGD(params=list(model.parameters()), lr=0.01, momentum=0.9)

    losses_log, batches_log = train_clustering(
        model, dl, loss, optimizer, device, epochs=30
    )

    save_reconstruction_results(
        "cluster",
        losses_log,
        batches_log,
        dl,
        model,
        device,
        dir="../models/DEC_small/",
    )


if __name__ == "__main__":
    np.random.seed(42)
    # reconstruction_experiment()
    clustering_experiment()
