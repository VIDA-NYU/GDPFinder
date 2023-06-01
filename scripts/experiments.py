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
    filenames = get_filenames()[:400000]
    dataset = get_sample_patches_dataset(filenames=filenames, resize=(28, 28))
    dl = DataLoader(dataset, batch_size=256, shuffle=True)

    print(f"Dataset shape: {len(dataset)}")
    print("Training AutoEncoder ...")
    print("===================================")

    model = SmallAutoEncoder(50, layers_per_block=4).to(device)

    print(
        f"Nº parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)/1000000:.2f}M"
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
        n_clusters=10, embedding_dim=50, encoder=encoder, cluster_centers=centers
    )
    model.to(device)

    print(
        f"Number of parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)/1000000:.2f}M"
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


def big_experiment():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    filenames = get_filenames()[:600000]
    filenames_test = get_filenames()[:100]
    dataset = get_sample_patches_dataset(filenames=filenames)
    dataset_test = get_sample_patches_dataset(filenames=filenames_test)
    dl = DataLoader(dataset, batch_size=96)
    dl_test = DataLoader(dataset_test, batch_size=96)

    print(f"Dataset shape: {len(dataset)}")
    print("Training AutoEncoder ...")
    print("===================================")

    latent_dim = 20
    model = AutoEncoder(
        latent_dim=latent_dim,
        encoder_arch="resnet50",
        encoder_lock_weights=True,
        decoder_layers_per_block=[3, 3, 3, 3, 3],
        decoder_enable_bn=False,
    ).to(device)

    print(
        f"Nº parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)/1000000:.2f}M"
    )

    loss = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    losses_log, batches_log = train_reconstruction(
        model,
        dl,
        loss,
        optimizer,
        device,
        epochs=40,
        test_loader=dl_test,
        dir="../models/AE_resnet50/",
    )
    save_reconstruction_results(
        "reconstruction",
        losses_log,
        batches_log,
        dl,
        model,
        device,
        dir="../models/AE_resnet50/",
    )

    print("Training Clustering AutoEncoder ...")
    print("===================================")
    print(f"Dataset shape: {len(dataset)}")

    model.eval()
    embeddings = get_embeddings(dl, model, device)
    centers = torch.tensor(cluster_embeddings(embeddings, 10))
    encoder = model.encoder

    model = DEC(
        n_clusters=10,
        embedding_dim=latent_dim,
        encoder=encoder,
        cluster_centers=centers,
    )
    model.to(device)

    print(
        f"Number of parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)/1000000:.2f}M"
    )

    loss = nn.KLDivLoss(size_average=False)

    optimizer = torch.optim.SGD(params=list(model.parameters()), lr=0.01, momentum=0.9)

    losses_log, batches_log = train_clustering(
        model,
        dl,
        loss,
        optimizer,
        device,
        epochs=40,
        test_loader=dl_test,
        dir="../models/DEC_resnet50/",
    )

    save_reconstruction_results(
        "cluster",
        losses_log,
        batches_log,
        dl,
        model,
        device,
        dir="../models/DEC_resnet50/",
    )


def big_experiment_v2():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    filenames = get_filenames()[:1000000]
    filenames_test = get_filenames()[:int(96*4)]
    dataset = get_sample_patches_dataset(filenames=filenames)
    dataset_test = get_sample_patches_dataset(filenames=filenames_test)
    dl = DataLoader(dataset, batch_size=96)
    dl_test = DataLoader(dataset_test, batch_size=96)

    print(f"Dataset shape: {len(dataset)}")
    print("Training AutoEncoder ...")
    print("===================================")

    latent_dim = 100
    model = AutoEncoder(
        latent_dim=latent_dim,
        encoder_arch="resnet50",
        encoder_lock_weights=False,
        decoder_layers_per_block=[3, 3, 3, 3, 3],
        decoder_enable_bn=False,
    ).to(device)

    print(
        f"Nº parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)/1000000:.2f}M"
    )

    loss = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    losses_log, batches_log = train_reconstruction(
        model,
        dl,
        loss,
        optimizer,
        device,
        epochs=4,
        test_loader=dl_test,
        dir="../models/AE_resnet50_v2/",
    )
    save_reconstruction_results(
        "reconstruction",
        losses_log,
        batches_log,
        dl,
        model,
        device,
        dir="../models/AE_resnet50_v2/",
    )


def varying_clusters_experiment():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    filenames_train = get_filenames()[:150000]
    filenames_test = get_filenames()[:1000]
    dataset_train = get_sample_patches_dataset(filenames=filenames_train)
    dataset_test = get_sample_patches_dataset(filenames=filenames_test)
    dl_train = DataLoader(dataset_train, batch_size=96)
    dl_test = DataLoader(dataset_test, batch_size=96)
    latent_dim = 20
    model = AutoEncoder(
        latent_dim=latent_dim,
        encoder_arch="resnet50",
        encoder_lock_weights=True,
        decoder_layers_per_block=[3, 3, 3, 3, 3],
        decoder_enable_bn=False,
    )
    model.load_state_dict(torch.load("../models/AE_resnet50/model.pt"))
    model = model.to(device)

    print(
        f"Nº parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)/1000000:.2f}M"
    )
    print("Training Clustering AutoEncoder ...")
    print("===================================")
    print(f"Dataset shape: {len(dataset_train)}")

    model.eval()
    encoder = model.encoder
    embeddings = get_embeddings(dl_train, model, device)

    for k in [10, 20, 30, 50, 100]:
        centers = torch.tensor(cluster_embeddings(embeddings, k))
        print(f"Training DEC with k={k}")
        model = DEC(
            n_clusters=k,
            embedding_dim=latent_dim,
            encoder=encoder,
            cluster_centers=centers,
        ).to(device)

        print(
            f"Number of parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)/1000000:.2f}M"
        )

        loss = nn.KLDivLoss(size_average=False)
        optimizer = torch.optim.SGD(
            params=list(model.parameters()), lr=0.01, momentum=0.9
        )

        losses_log, batches_log = train_clustering(
            model,
            dl_train,
            loss,
            optimizer,
            device,
            epochs=25,
            test_loader=dl_test,
            dir=f"../models/DEC_resnet50_clusters_{k}/",
        )

        save_reconstruction_results(
            "cluster",
            losses_log,
            batches_log,
            dl_train,
            model,
            device,
            dir=f"../models/DEC_resnet50_clusters_{k}/",
        )

def varying_clusters_experiment_v2():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #filenames_train = get_filenames()[:1000000]
    #filenames_test = get_filenames()[:1000]
    filenames_train = os.listdir("../data/output/ny_patches")
    filenames_train = [os.path.join("../data/output/patches", f) for f in filenames_train if f.endswith(".png")]
    filenames_test = filenames_train[:1000]
    dataset_train = get_sample_patches_dataset(filenames=filenames_train)
    dataset_test = get_sample_patches_dataset(filenames=filenames_test)
    dl_train = DataLoader(dataset_train, batch_size=96)
    dl_test = DataLoader(dataset_test, batch_size=96)
    latent_dim = 100
    model = AutoEncoder(
        latent_dim=latent_dim,
        encoder_arch="resnet50",
        encoder_lock_weights=False,
        decoder_layers_per_block=[3, 3, 3, 3, 3],
        decoder_enable_bn=False,
    )
    model.load_state_dict(torch.load("../models/AE_resnet50_v2/model.pt"))
    model = model.to(device)

    print(
        f"Nº parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)/1000000:.2f}M"
    )
    print("Training Clustering AutoEncoder ...")
    print("===================================")
    print(f"Dataset shape: {len(dataset_train)}")

    model.eval()
    encoder = model.encoder
    embeddings = get_embeddings(dl_train, model, device)

    for k in [10, 20, 30, 50]:
        centers = torch.tensor(cluster_embeddings(embeddings, k))
        print(f"Training DEC with k={k}")
        model = DEC(
            n_clusters=k,
            embedding_dim=latent_dim,
            encoder=encoder,
            cluster_centers=centers,
        ).to(device)

        print(
            f"Number of parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)/1000000:.2f}M"
        )

        loss = nn.KLDivLoss(size_average=False)
        optimizer = torch.optim.SGD(
            params=list(model.parameters()), lr=0.01, momentum=0.9
        )

        losses_log, batches_log = train_clustering(
            model,
            dl_train,
            loss,
            optimizer,
            device,
            epochs=10,
            test_loader=dl_test,
            dir=f"../models/DEC_resnet50_clusters_{k}_v2/",
        )

        save_reconstruction_results(
            "cluster",
            losses_log,
            batches_log,
            dl_train,
            model,
            device,
            dir=f"../models/DEC_resnet50_clusters_{k}_v2/",
        )



if __name__ == "__main__":
    np.random.seed(42)
    # small_experiment()
    # big_experiment()
    # varying_clusters_experiment()
    # big_experiment_v2()
    varying_clusters_experiment_v2()
