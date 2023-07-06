import os
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from tqdm import tqdm

import data
import models
from train import (
    train_reconstruction,
    train_clustering,
    train_reconstruction_feature_extraction,
)
from utils import save_reconstruction_results, get_embeddings, cluster_embeddings


def small_experiment():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    filenames = data.get_filenames()[:400000]
    dataset = data.get_sample_patches_dataset(filenames=filenames, resize=(28, 28))
    dl = DataLoader(dataset, batch_size=256, shuffle=True)

    print(f"Dataset shape: {len(dataset)}")
    print("Training AutoEncoder ...")
    print("===================================")

    model = models.SmallAutoEncoder(50, layers_per_block=4).to(device)

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

    model = models.DEC(
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
    filenames = data.get_filenames(2250)
    filenames_test = filenames[:96]
    dataset = data.SmallPatchesDataset(filenames, resnet=False)
    dataset_test = data.SmallPatchesDataset(filenames_test, resnet=False)
    dl = DataLoader(dataset, batch_size=96)
    dl_test = DataLoader(dataset_test, batch_size=96)

    print(f"Dataset shape: {len(dataset)}")
    print("Training AutoEncoder ...")
    print("===================================")

    latent_dim = 100
    model = models.AutoEncoder(
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
        epochs=10,
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


def varying_clusters_experiment():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    filenames_train = data.get_filenames(1000)
    filenames_test = filenames_train[:96]
    dataset_train = data.SmallPatchesDataset(filenames_train, resnet=False)
    dataset_test = data.SmallPatchesDataset(filenames_test, resnet=False)
    dl_train = DataLoader(dataset_train, batch_size=96)
    dl_test = DataLoader(dataset_test, batch_size=96)
    latent_dim = 100
    model = models.AutoEncoder(
        latent_dim=latent_dim,
        encoder_arch="resnet50",
        encoder_lock_weights=False,
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

    for k in [10, 20, 30, 50]:
        centers = torch.tensor(cluster_embeddings(embeddings, k))
        print(f"Training DEC with k={k}")
        model = models.DEC(
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


def varying_latent_dim():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    filenames = data.get_filenames(1000)
    filenames_test = filenames[:96]
    dataset = data.SmallPatchesDataset(filenames, resnet=False)
    dataset_test = data.SmallPatchesDataset(filenames_test, resnet=False)
    dl = DataLoader(dataset, batch_size=96)
    dl_test = DataLoader(dataset_test, batch_size=96)

    print(f"Dataset shape: {len(dataset)}")
    print("Training AutoEncoder ...")
    print("===================================")

    for latent_dim in [10, 32, 64, 128, 256, 512]:
        model = models.AutoEncoder(
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
            epochs=10,
            test_loader=dl_test,
            dir=f"../models/AE_resnet50_{latent_dim}/",
        )
        save_reconstruction_results(
            "reconstruction",
            losses_log,
            batches_log,
            dl,
            model,
            device,
            dir=f"../models/AE_resnet50_{latent_dim}/",
        )


def varying_latent_dim_small():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    filenames_train = data.get_filenames_small_patches(1000)
    filenames_train, filenames_test = train_test_split(
        filenames_train, test_size=0.1, shuffle=True, random_state=0
    )
    dataset_train = data.SmallPatchesDataset(filenames_train)
    dataset_test = data.SmallPatchesDataset(filenames_test)
    dl_train = DataLoader(dataset_train, batch_size=96)
    dl_test = DataLoader(dataset_test, batch_size=192)

    print(f"Train samples: {len(dataset_train)} \t Test samples: {len(dataset_test)}")
    print("Training AutoEncoder ...")
    print("===========================================================")

    for latent_dim in [32, 64, 128, 256, 512]:
        model = models.AutoEncoder(
            latent_dim=latent_dim,
            encoder_arch="resnet50_small_patch",
            encoder_lock_weights=False,
            decoder_layers_per_block=[3, 3, 3, 3, 3],
            denoising=True,
        ).to(device)
        n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Nº parameters: {n_parameters/1000000:.2f}M")

        loss = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        train_reconstruction(
            model,
            dl_train,
            dl_test,
            loss,
            optimizer,
            device,
            epochs=10,
            dir=f"../models/AE_resnet50_small_{latent_dim}/",
        )


def varying_clusters_small(latent_dim=128):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    filenames_train = data.get_filenames_center_blocks()
    filenames_train, filenames_test = train_test_split(
        filenames_train, test_size=0.1, shuffle=True, random_state=0
    )
    dataset_train = data.SmallPatchesDataset(filenames_train)
    dataset_test = data.SmallPatchesDataset(filenames_test)
    dl_train = DataLoader(dataset_train, batch_size=96)
    dl_test = DataLoader(dataset_test, batch_size=192)
    model = models.AutoEncoder(
        latent_dim=latent_dim,
        encoder_arch="resnet50_small_patch",
        encoder_lock_weights=False,
        decoder_layers_per_block=[3, 3, 3, 3, 3],
        denoising=True,
    )
    model.load_state_dict(
        torch.load(f"../models/AE_resnet50_small_{latent_dim}/model.pt")
    )
    model = model.to(device)

    print(
        f"Nº parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)/1000000:.2f}M"
    )
    print("Training Clustering AutoEncoder ...")
    print("===================================")
    print(f"Dataset shape: {len(dataset_train)}")

    model.eval()
    encoder = model.encoder
    embeddings = []
    with torch.no_grad():
        for batch in tqdm(dl_train):
            batch = batch.to(device)
            embeddings.append(
                encoder(batch).detach().cpu().numpy().reshape(batch.shape[0], -1)
            )
    embeddings = np.concatenate(embeddings)

    for k in [10, 20, 30, 50]:
        centers = torch.tensor(cluster_embeddings(embeddings, k))
        print(f"Training DEC with k={k}")
        model = models.DEC(
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

        train_clustering(
            model,
            dl_train,
            dl_test,
            loss,
            optimizer,
            device,
            epochs=5,
            dir=f"../models/DEC_resnet50_clusters_{k}_latent_dim_{latent_dim}_small/",
        )


def experiment_clustering_fixed_k(k):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    filenames_train = data.get_filenames(1000)
    filenames_test = filenames_train[:96]
    dataset_train = data.SmallPatchesDataset(filenames_train, resnet=False)
    dataset_test = data.SmallPatchesDataset(filenames_test, resnet=False)
    dl_train = DataLoader(dataset_train, batch_size=96)
    dl_test = DataLoader(dataset_test, batch_size=96)
    latent_dim = 100
    model = models.AutoEncoder(
        latent_dim=latent_dim,
        encoder_arch="resnet50",
        encoder_lock_weights=False,
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
    centers = torch.tensor(cluster_embeddings(embeddings, k))
    model = models.DEC(
        n_clusters=k,
        embedding_dim=latent_dim,
        encoder=encoder,
        cluster_centers=centers,
    ).to(device)

    print(
        f"Number of parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)/1000000:.2f}M"
    )

    loss = nn.KLDivLoss(size_average=False)
    optimizer = torch.optim.SGD(params=list(model.parameters()), lr=0.01, momentum=0.9)

    losses_log, batches_log = train_clustering(
        model,
        dl_train,
        loss,
        optimizer,
        device,
        epochs=10,
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


def denoising_varying_latent_dim():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    filenames = data.get_filenames(1000)
    filenames_test = filenames[:96]
    dataset = data.SmallPatchesDataset(filenames, resnet=False)
    dataset_test = data.SmallPatchesDataset(filenames_test, resnet=False)
    dl = DataLoader(dataset, batch_size=96)
    dl_test = DataLoader(dataset_test, batch_size=96)

    print(f"Dataset shape: {len(dataset)}")
    print("Training AutoEncoder ...")
    print("===================================")

    for latent_dim in [10, 32, 64, 128, 256, 512]:
        model = models.DenoisingAutoEncoder(
            latent_dim=latent_dim,
            layers_per_block=[3, 3, 3, 3, 3],
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
            epochs=10,
            test_loader=dl_test,
            dir=f"../models/DAE_{latent_dim}/",
        )
        save_reconstruction_results(
            "reconstruction",
            losses_log,
            batches_log,
            dl,
            model,
            device,
            dir=f"../models/DAE_{latent_dim}/",
        )


def varying_latent_dim_resnet_extractor():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    filenames_train = data.get_filenames_center_blocks()
    filenames_train, filenames_test = train_test_split(
        filenames_train, test_size=0.1, shuffle=True, random_state=0
    )
    dataset_train = data.SmallPatchesDataset(filenames_train, resize=(224, 224))
    dataset_test = data.SmallPatchesDataset(filenames_test, resize=(224, 224))
    dl_train = DataLoader(dataset_train, batch_size=300)
    dl_test = DataLoader(dataset_test, batch_size=300)

    print(f"Train samples: {len(dataset_train)} \t Test samples: {len(dataset_test)}")
    print("Training AutoEncoder ...")
    print("===========================================================")

    for latent_dim in [
        # 64,
        # 128,
        256
    ]:
        model = models.AutoEncoderResnetExtractor(latent_dim).to(device)
        n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Nº parameters: {n_parameters/1000000:.2f}M")

        loss = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        train_reconstruction_feature_extraction(
            model,
            dl_train,
            dl_test,
            loss,
            optimizer,
            device,
            epochs=10,
            dir=f"../models/AE_extractor_resnet50_small_{latent_dim}/",
        )


def varying_clusters_resnet_extractor(latent_dim=128):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    filenames_train = data.get_filenames_center_blocks()
    filenames_train, filenames_test = train_test_split(
        filenames_train, test_size=0.1, shuffle=True, random_state=0
    )
    dataset_train = data.SmallPatchesDataset(filenames_train, resize=(224, 224))
    dataset_test = data.SmallPatchesDataset(filenames_test, resize=(224, 224))
    dl_train = DataLoader(dataset_train, batch_size=300)
    dl_test = DataLoader(dataset_test, batch_size=300)
    model = models.AutoEncoderResnetExtractor(latent_dim)
    model.load_state_dict(
        torch.load(f"../models/AE_extractor_resnet50_small_{latent_dim}/model.pt")
    )
    model = model.to(device)

    print(
        f"Nº parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)/1000000:.2f}M"
    )
    print("Training Clustering AutoEncoder ...")
    print("===================================")
    print(f"Dataset shape: {len(dataset_train)}")

    model.eval()
    encoder = model.encoder
    embeddings = []
    with torch.no_grad():
        for batch in tqdm(dl_train):
            batch = batch.to(device)
            embeddings.append(
                encoder(batch).detach().cpu().numpy().reshape(batch.shape[0], -1)
            )
    embeddings = np.concatenate(embeddings)

    for k in [
        # 10,
        # 20,
        30,
        50,
    ]:
        centers = torch.tensor(cluster_embeddings(embeddings, k))
        print(f"Training DEC with k={k}")
        model = models.DEC(
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

        train_clustering(
            model,
            dl_train,
            dl_test,
            loss,
            optimizer,
            device,
            epochs=5,
            dir=f"../models/DEC_resnet50_clusters_{k}_latent_dim_{latent_dim}_feature_extractor/",
        )


if __name__ == "__main__":
    np.random.seed(42)
    # small_experiment()
    # big_experiment()
    # varying_clusters_experiment()
    # experiment_clustering_fixed_k(k=2)
    # varying_latent_dim()
    # denoising_varying_latent_dim()
    # varying_latent_dim_small()
    # varying_clusters_small()
    # varying_clusters_small(512)
    # varying_latent_dim_resnet_extractor()
    varying_clusters_resnet_extractor(128)
