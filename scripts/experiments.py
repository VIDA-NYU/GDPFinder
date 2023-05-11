import os
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader

from data import get_sample_patches_dataset, get_filenames
from models import AutoEncoder, SmallAutoEncoder, DEC
from train import train_reconstruction, train_clustering
from utils import save_reconstruction_results, get_embeddings, cluster_embeddings

def small_experiment():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    filenames = get_filenames()[:200000]
    dataset = get_sample_patches_dataset(filenames=filenames, resize=(28, 28))
    dl = DataLoader(dataset, batch_size=256, shuffle=True)

    print("Training AutoEncoder ...")
    print("===================================")
    print(f"Dataset shape: {len(dataset)}")

    model = SmallAutoEncoder(20, layers_per_block=4).to(device)

    print(
        f"Nº parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)//1000000:2f}M"
    )

    loss = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    losses_log, batches_log = train_reconstruction(
        model, dl, loss, optimizer, device, epochs=15
    )
    save_reconstruction_results(
        "reconstruction",
        losses_log,
        batches_log,
        dl,
        model,
        device,
        dir="../models/AE_medium/",
    )

    print("Training Clustering AutoEncoder ...")
    print("===================================")
    print(f"Dataset shape: {len(dataset)}")

    model.eval()
    embeddings = get_embeddings(dl, model, device)
    centers = torch.tensor(cluster_embeddings(embeddings, 10))
    encoder = model.encoder

    model = DEC(
        n_clusters=10, embedding_dim=20, encoder=encoder, cluster_centers=centers
    )
    model.to(device)

    print(
        f"Number of parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)//1000000:.2f}M"
    )

    loss = nn.KLDivLoss(size_average=False)

    optimizer = torch.optim.SGD(params=list(model.parameters()), lr=0.01, momentum=0.9)

    losses_log, batches_log = train_clustering(
        model, dl, loss, optimizer, device, epochs=15
    )

    save_reconstruction_results(
        "cluster",
        losses_log,
        batches_log,
        dl,
        model,
        device,
        dir="../models/DEC_medium/",
    )


if __name__ == "__main__":
    np.random.seed(42)
    small_experiment()
