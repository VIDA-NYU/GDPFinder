import torch
import numpy as np
from models import target_distribution
from utils import save_reconstruction_results, save_clustering_results
from tqdm import tqdm


def train_reconstruction(
    model,
    dl_train,
    dl_test=None,
    loss=None,
    optimizer=None,
    device=None,
    epochs=100,
    return_embeddings=False,
    dir=None,
    verbose=True,
):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if loss is None:
        loss = torch.nn.MSELoss()
    if optimizer is None:
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    train_losses_log = []
    train_batch_losses_log = []
    test_losses_log = []
    for i in range(epochs):
        if return_embeddings:
            embeddings = []
        iter_loss = 0
        n = 0
        for batch in tqdm(dl_train):
            batch = batch.to(device)
            encoded = model.encoder(batch)
            decoded = model.decoder(encoded)
            rec_loss = loss(decoded, batch)
            optimizer.zero_grad()
            rec_loss.backward()
            optimizer.step()
            iter_loss += rec_loss.item()
            n += batch.shape[0]
            train_batch_losses_log.append(rec_loss.item())
            if return_embeddings:
                embeddings.append(encoded.detach().cpu().numpy())
        train_losses_log.append(iter_loss / n)
        train_loss = iter_loss

        if dl_test is not None:
            with torch.no_grad():
                model.eval()
                iter_loss = 0
                n = 0
                for batch in tqdm(dl_test):
                    batch = batch.to(device)
                    decoded = model(batch)
                    rec_loss = loss(decoded, batch)
                    iter_loss += rec_loss.item()
                    n += batch.shape[0]
                test_losses_log.append(iter_loss / n)

            save_reconstruction_results(
                model,
                train_losses_log,
                train_batch_losses_log,
                test_losses_log,
                batch,
                decoded,
                dir=dir,
            )
            model.train()

        if verbose:
            print(f"Epoch {i+1}/{epochs} - Loss: {train_loss:.8f}")

    if return_embeddings:
        return np.concatenate(embeddings, axis=0)


def train_clustering(
    model,
    dl_train,
    dl_test = None,
    loss = torch.nn.KLDivLoss(reduction="sum"),
    optimizer = None,
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    epochs=100,
    return_clusters=False,
    dir=None,
    verbose=True,
):
    train_losses_log = []
    train_batch_losses_log = []
    test_losses_log = []
    if optimizer is None:
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)

    for i in range(epochs):
        if return_clusters:
            clusters = []
        iter_loss = 0
        n = 0
        for batch in tqdm(dl_train):
            batch = batch.to(device)
            output = model(batch)
            target = target_distribution(output).detach()
            cluster_loss = loss(output.log(), target) / output.shape[0]
            optimizer.zero_grad()
            cluster_loss.backward()
            optimizer.step()
            iter_loss += cluster_loss.item()
            n += batch.shape[0]
            train_batch_losses_log.append(cluster_loss.item())
            if return_clusters:
                clusters.append(output.detach().argmax(dim=1).cpu().numpy())
        train_losses_log.append(iter_loss / n)

        if dl_test is not None:
            with torch.no_grad():
                model.eval()
                iter_loss_test = 0
                for batch in tqdm(dl_test):
                    batch = batch.to(device)
                    output = model(batch)
                    target = target_distribution(output).detach()
                    cluster_loss = loss(output.log(), target) / output.shape[0]
                    iter_loss_test += cluster_loss.item()
                    n += batch.shape[0]
                test_losses_log.append(iter_loss_test / n)

            save_clustering_results(
                model,
                train_losses_log,
                train_batch_losses_log,
                test_losses_log,
                dir=dir,
            )

            model.train()
        if verbose:
            print(f"Epoch {i+1}/{epochs} - Loss: {iter_loss:.8f}")

    if return_clusters:
        return np.concatenate(clusters, axis=0)


def train_reconstruction_feature_extraction(
    model,
    dl_train,
    dl_test=None,
    loss=torch.nn.MSELoss(),
    optimizer=None,
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    epochs=100,
    return_embeddings=False,
    dir=None,
    verbose=True,
):
    if optimizer is None:
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    train_losses_log = []
    train_batch_losses_log = []
    test_losses_log = []
    for i in range(epochs):
        if return_embeddings:
            embeddings = []
        iter_loss = 0
        n = 0
        for batch in tqdm(dl_train):
            batch = batch.to(device)
            if return_embeddings:
                encoded = model.encoder(batch)
                embeddings.append(encoded.detach().cpu().numpy())
            feature, decoded = model(batch)
            rec_loss = loss(decoded, feature)
            optimizer.zero_grad()
            rec_loss.backward()
            optimizer.step()
            iter_loss += rec_loss.item()
            n += batch.shape[0]
            train_batch_losses_log.append(rec_loss.item())
        train_losses_log.append(iter_loss / n)

        train_loss = iter_loss

        if dl_test is not None:
            with torch.no_grad():
                model.eval()
                iter_loss = 0
                n = 0
                for batch in tqdm(dl_test):
                    batch = batch.to(device)
                    feature, decoded = model(batch)
                    rec_loss = loss(decoded, feature)
                    iter_loss += rec_loss.item()
                    n += batch.shape[0]
                test_losses_log.append(iter_loss / n)

            save_reconstruction_results(
                model,
                train_losses_log,
                train_batch_losses_log,
                test_losses_log,
                batch,
                decoded,
                dir=dir,
            )

            model.train()
        if verbose:
            print(f"Epoch {i+1}/{epochs} - Loss: {train_loss:.8f}")

    if return_embeddings:
        return np.concatenate(embeddings, axis=0)
