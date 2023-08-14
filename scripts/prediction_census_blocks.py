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


def load_blocks_df(patches_count_max = 50):
    blocks_train = pd.read_csv(f"../data/blocks_patches_relation_train_{patches_count_max}.csv")
    blocks_val = pd.read_csv(f"../data/blocks_patches_relation_val_{patches_count_max}.csv")
    blocks_test = pd.read_csv(f"../data/blocks_patches_relation_test_{patches_count_max}.csv")
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
        "n_estimators": [10, 25, 50],
        "max_depth": [10, 20, 20],
        "min_samples_leaf": [1, 5, 10],
        # "max_features": ["auto", "sqrt", "log2"]
    }
    clf = GridSearchCV(rf, parameters, n_jobs=-1, verbose=2)
    clf.fit(x_train, y_train)
    r2_train, r2_test, mae_train, mae_test = eval(clf, x_train, y_train, x_test, y_test)
    return r2_train, r2_test, mae_train, mae_test, clf


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
    print("Evaluation of clustering model")
    print("Clustering the patches of each block")
    blocks_train, blocks_val, blocks_test = load_blocks_df()
    blocks_train = cluster_patches(blocks_train, model)
    blocks_val = cluster_patches(blocks_val, model)
    blocks_test = cluster_patches(blocks_test, model)

    results = []
    for method in ["fraction", "distance"]:
        if method == "fraction":
            blocks_train = get_fraction_features(blocks_train, k)
            blocks_val = get_fraction_features(blocks_val, k)
            blocks_test = get_fraction_features(blocks_test, k)
        elif method == "distance":
            blocks_train = get_distance_features(blocks_train, k)
            blocks_val = get_distance_features(blocks_val, k)
            blocks_test = get_distance_features(blocks_test, k)

        columns = blocks_train.columns.str.contains("feature_")
        x_train = blocks_train.loc[:, columns].values
        x_val = blocks_val.loc[:, columns].values
        x_test = blocks_test.loc[:, columns].values
        for target in ["mhi", "density", "ed_attain"]:
            print(f"Fitting model for {target} with {method} features")
            y_train = blocks_train[target].values
            y_val = blocks_val[target].values
            y_test = blocks_test[target].values
            r2_train, r2_val, mae_train, mae_val, clf = grid_search_rf(
                x_train, y_train, x_val, y_val
            )
            r2_test, mae_test, _, _ = eval(clf, x_test, y_test, x_test, y_test)
            results.append(
                {
                    "method": method,
                    "target": target,
                    "r2_train": r2_train,
                    "r2_val": r2_val,
                    "mae_train": mae_train,
                    "mae_val": mae_val,
                    "r2_test": r2_test,
                    "mae_test": mae_test,
                }
            )
    return pd.DataFrame(results)
