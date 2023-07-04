import torch
from models import target_distribution
from utils import save_reconstruction_results, save_clustering_results
from tqdm import tqdm


def train_reconstruction(
    model,
    dl_train,
    dl_test,
    loss,
    optimizer,
    device,
    epochs=100,
    dir=None,
    verbose=True,
):
    train_losses_log = []
    train_batch_losses_log = []
    test_losses_log = []
    for i in range(epochs):
        iter_loss = 0
        n = 0
        for batch in tqdm(dl_train):
            batch = batch.to(device)
            decoded = model(batch)
            rec_loss = loss(decoded, batch)
            optimizer.zero_grad()
            rec_loss.backward()
            optimizer.step()
            iter_loss += rec_loss.item()
            n += batch.shape[0]
            train_batch_losses_log.append(rec_loss.item())
        train_losses_log.append(iter_loss / n)

        if verbose:
            print(f"Epoch {i+1}/{epochs} - Loss: {iter_loss:.8f}")

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


def train_clustering(
    model,
    dl_train,
    dl_test,
    loss,
    optimizer,
    device,
    epochs=100,
    dir=None,
    verbose=True,
):
    train_losses_log = []
    train_batch_losses_log = []
    test_losses_log = []

    for i in range(epochs):
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
        train_losses_log.append(iter_loss / n)

        if verbose:
            print(f"Epoch {i+1}/{epochs} - Loss: {iter_loss:.8f}")

        with torch.no_grad():
            model.eval()
            iter_loss = 0
            for batch in tqdm(dl_test):
                batch = batch.to(device)
                output = model(batch)
                target = target_distribution(output).detach()
                cluster_loss = loss(output.log(), target) / output.shape[0]
                iter_loss += cluster_loss.item()
                n += batch.shape[0]
            test_losses_log.append(iter_loss / n)

            save_clustering_results(
                model,
                train_losses_log,
                train_batch_losses_log,
                test_losses_log,
                dir=dir,
            )
        model.train()

def train_reconstruction_feature_extraction(
    model,
    dl_train,
    dl_test,
    loss,
    optimizer,
    device,
    epochs=100,
    dir=None,
    verbose=True,
):
    train_losses_log = []
    train_batch_losses_log = []
    test_losses_log = []
    for i in range(epochs):
        iter_loss = 0
        n = 0
        for batch in tqdm(dl_train):
            batch = batch.to(device)
            feature, decoded = model(batch)
            rec_loss = loss(decoded, feature)
            optimizer.zero_grad()
            rec_loss.backward()
            optimizer.step()
            iter_loss += rec_loss.item()
            n += batch.shape[0]
            train_batch_losses_log.append(rec_loss.item())
        train_losses_log.append(iter_loss / n)

        if verbose:
            print(f"Epoch {i+1}/{epochs} - Loss: {iter_loss:.8f}")

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
