import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler
from ast import literal_eval
import joblib
from tqdm import tqdm

import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

import models
import data
import utils


def load_blocks_df():
    blocks_train = pd.read_csv("../data/blocks_patches_relation_train.csv")
    blocks_val = pd.read_csv("../data/blocks_patches_relation_val.csv")
    blocks_test = pd.read_csv("../data/blocks_patches_relation_test.csv")
    blocks_train["filenames"] = blocks_train["filenames"].apply(literal_eval)
    blocks_val["filenames"] = blocks_val["filenames"].apply(literal_eval)
    blocks_test["filenames"] = blocks_test["filenames"].apply(literal_eval)
    return blocks_train, blocks_val, blocks_test


def cluster_patches(blocks_df, model):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    clusters = []
    clusters_distance = []
    for i, row in tqdm(blocks_df.iterrows(), total=len(blocks_df)):
        filenames = row.filenames
        dataset = data.SmallPatchesDataset(filenames)
        dl = DataLoader(dataset, batch_size=500, shuffle=False)
        clusters.append(utils.get_clusters(dl, model))
        clusters_distance.append(utils.get_clusters_distances(dl, model))

    blocks_df["clusters"] = clusters
    blocks_df["clusters_distance"] = clusters_distance
    return blocks_df


def get_model(latent_dim, k, method):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    kmeans = joblib.load(f"../models/AE_extractor_resnet50_64/kmeans_{k}_clusters.pkl")
    model = models.AutoEncoderResnetExtractor(dims=[2048, 64])
    model.load_state_dict(torch.load("../models/AE_extractor_resnet50_64/model.pt"))
    model_dec = models.DEC(
        n_clusters=k,
        embedding_dim=64,
        encoder=model.encoder,
        cluster_centers=torch.tensor(kmeans.cluster_centers_),
    )
    model_dec.to(device)
    model_dec.eval()
    return model_dec


def get_clusters_patches(model, filenames):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = data.SmallPatchesDataset(filenames)
    dl = DataLoader(dataset, batch_size=1000, shuffle=False)
    clusters = []
    clusters_distance = []
    with torch.no_grad():
        for x in dl:
            x = x.to(device)
            c = model(x)
            clusters.append(c.cpu().numpy())
            d = model.centroids_distance(x)
            clusters_distance.append(d.cpu().numpy())
    clusters = np.concatenate(clusters)
    clusters = clusters.argmax(axis=1)
    clusters_distance = np.concatenate(clusters_distance)
    return clusters, clusters_distance


def get_fraction_features(blocks_df, k):
    blocks_df = blocks_df.loc[:, ~blocks_df.columns.str.contains("feature_")]
    data = np.zeros((blocks_df.shape[0], k))
    for i, (_, row) in enumerate(blocks_df.iterrows()):
        clusters = row.clusters
        x = np.bincount(clusters, minlength=k).astype(float)
        x /= x.sum()
        data[i, :] = x
    data = pd.DataFrame(data, columns=[f"feature_{i}" for i in range(k)])
    blocks_df = pd.concat(
        [blocks_df.reset_index(drop=True), data.reset_index(drop=True)], axis=1
    )
    return blocks_df


def get_distance_features(blocks_df, k):
    blocks_df = blocks_df.loc[:, ~blocks_df.columns.str.contains("feature_")]
    data = np.zeros((blocks_df.shape[0], k))
    for i, (_, row) in enumerate(blocks_df.iterrows()):
        clusters_distance = row.clusters_distance
        x = np.mean(clusters_distance, axis=0)
        data[i, :] = x
    data = pd.DataFrame(data, columns=[f"feature_{i}" for i in range(k)])
    blocks_df = pd.concat(
        [blocks_df.reset_index(drop=True), data.reset_index(drop=True)], axis=1
    )
    return blocks_df


def eval(clf, x_train, y_train, x_test, y_test):
    mae_train = mean_absolute_error(y_train, clf.predict(x_train))
    r2_train = r2_score(y_train, clf.predict(x_train))

    mae_test = mean_absolute_error(y_test, clf.predict(x_test))
    r2_test = r2_score(y_test, clf.predict(x_test))
    return r2_train, r2_test, mae_train, mae_test


def grid_search_rf(x_train, y_train, x_test, y_test):
    rf = RandomForestRegressor()
    parameters = {
        "n_estimators": [10, 100, 1000],
        "max_depth": [10, 100],
        # "min_samples_split": [2, 10, 100],
    }
    clf = GridSearchCV(rf, parameters, n_jobs=-1)
    clf.fit(x_train, y_train)
    return eval(clf, x_train, y_train, x_test, y_test)


class MLP(nn.Module):
    def __init__(self, dims):
        super(MLP, self).__init__()
        self.layers = []
        for in_dim, out_dim in zip(dims[:-1], dims[1:]):
            self.layers.append(nn.Linear(in_dim, out_dim))
            if out_dim != dims[-1]:
                self.layers.append(nn.ReLU())
        self.layers = nn.Sequential(*self.layers)

    def forward(self, x):
        return self.layers(x)

    def predict(self, x):
        device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        self.eval()
        if type(x) == pd.DataFrame:
            x_ = torch.from_numpy(x.values)
        elif type(x) == np.ndarray:
            x_ = torch.from_numpy(x)

        with torch.no_grad():
            x_ = x_.to(device)
            y = self.layers(x_)
            return y.detach().cpu().numpy()


def train_mlp(model, dl_train, dl_test):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    test_loss = []
    for i in range(100):
        iter_loss = 0
        for x, y in dl_train:
            x, y = x.to(device), y.to(device)
            y_pred = model(x)
            loss = criterion(y_pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            iter_loss += loss.item()

        if i % 3 == 0:
            iter_loss = 0
            with torch.no_grad():
                for x, y in dl_test:
                    x, y = x.to(device), y.to(device)
                    y_pred = model(x)
                    loss = criterion(y_pred, y)
                    iter_loss += loss.item()
                test_loss.append(iter_loss)

            if i > 10 and test_loss[-1] > test_loss[-2]:
                break


def grid_search_mlp(x_train, y_train, x_test, y_test):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    idx_train_, idx_val = train_test_split(
        np.arange(x_train.shape[0]), test_size=0.2, random_state=0
    )
    x_val_, y_val_ = x_train.values[idx_val, :], y_train[idx_val]
    x_train_, y_train_ = x_train.values[idx_train_, :], y_train[idx_train_]
    x_test_, y_test_ = x_test.values, y_test
    scaler = StandardScaler()
    x_train_ = scaler.fit_transform(x_train_)
    x_val_ = scaler.transform(x_val_)
    x_test_ = scaler.transform(x_test_)
    dl_train_ = DataLoader(
        TensorDataset(torch.tensor(x_train_), torch.tensor(y_train_.reshape(-1, 1))),
        batch_size=128,
    )
    dl_val = DataLoader(
        TensorDataset(torch.tensor(x_val_), torch.tensor(y_val_).reshape(-1, 1)),
        batch_size=128,
    )

    best_r2 = -np.inf
    best_model = None
    for dims in [
        [x_train.shape[1], 32, 64, 32, 1],
        [x_train.shape[1], 64, 256, 32, 1],
        [x_train.shape[1], 64, 512, 128, 1],
    ]:
        model_1 = MLP(dims)
        model_1.to(device, dtype=torch.double)
        train_mlp(model_1, dl_train_, dl_val)
        r2_train, r2_test = eval(model_1, x_train_, y_train_, x_val_, y_val_)

        if r2_test > best_r2:
            best_r2 = r2_test
            best_model = model_1

    return eval(best_model, x_train_, y_train_, x_test_, y_test_)


def eval_model(model, k):
    blocks_train, blocks_val, _ = load_blocks_df()  
    blocks_train = cluster_patches(blocks_train, model)
    blocks_val = cluster_patches(blocks_val, model) 

    results = []
    for method in ["fraction", "distance"]:
        if method == "fraction":
            blocks_train = get_fraction_features(blocks_train, k)
            blocks_val = get_fraction_features(blocks_val, k)
        elif method == "distance":
            blocks_train = get_distance_features(blocks_train, k)
            blocks_val = get_distance_features(blocks_val, k)

        columns = blocks_train.columns.str.contains("feature_")
        x_train = blocks_train.loc[:, columns].values
        x_val = blocks_val.loc[:, columns].values
        for target in ["mhi", "density", "ed_attain"]:
            y_train = blocks_train[target].values   
            y_val = blocks_val[target].values
            r2_train, r2_val, mae_train, mae_val = grid_search_mlp(
                x_train, y_train, x_val, y_val
            )
            results.append(
                {
                    "method": method,
                    "target": target,
                    "r2_train": r2_train,
                    "r2_val": r2_val,
                    "mae_train": mae_train,
                    "mae_val": mae_val,
                }
            )
    return pd.DataFrame(results)



if __name__ == "__main__":
    blocks_train, blocks_val, blocks_test = load_blocks_df(patches_count_max=100)
    blocks_train = blocks_train.sample(1000)
    blocks_val = blocks_val.sample(100)
    blocks_test = blocks_test.sample(100)

    latent_dim = 64
    for k in [20, 50, 100, 200]:
        print(k)
        model = get_model(latent_dim, k, method="kmeans")
        blocks_train = cluster_patches(blocks_train, model)
        blocks_val = cluster_patches(blocks_val, model)

        for method in ["fraction", "distance"]:
            print(method)
            if method == "fraction":
                blocks_train = get_fraction_features(blocks_train, k)
                blocks_val = get_fraction_features(blocks_val, k)
            elif method == "distance":
                blocks_train = get_distance_features(blocks_train, k)
                blocks_val = get_distance_features(blocks_val, k)

            for target in ["mhi", "density", "ed_attain"]:
                print(target)
                x_train = blocks_train.loc[
                    :, blocks_train.columns.str.contains("feature_")
                ].values
                y_train = blocks_train[target].values
                x_val = blocks_val.loc[
                    :, blocks_val.columns.str.contains("feature_")
                ].values
                y_val = blocks_val[target].values

                print(grid_search_rf(x_train, y_train, x_val, y_val))
