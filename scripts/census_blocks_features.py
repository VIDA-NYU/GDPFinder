import numpy as np
import pandas as pd
import geopandas as gpd
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler


import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

import models
import data


def load_blocks_data(intersection_threshold=0.25, patches_count_max=50):
    blocks_df = gpd.read_file("../data/census_blocks_patches_v2.geojson")

    # Clean data
    blocks_df = blocks_df[blocks_df.mhi > 0]
    blocks_df = blocks_df.dropna()
    blocks_df = blocks_df[blocks_df.patches_relation.apply(len) > 0]
    blocks_df["area_km2"] = blocks_df.geometry.to_crs({"proj": "cea"}).area / 10**6
    blocks_df["densisty"] = blocks_df["pop"] / blocks_df["area_km2"]

    def clean_patches_relation(s):
        s = s.split("\n")
        s = dict([x.split(":") for x in s])
        filenames = []
        data = []
        # transform test into array, filtering by intersection threshold
        for key, value in s.items():
            idx, ratio = value.split(" ")
            idx = np.array([float(v) for v in idx.split(",")])
            ratio = np.array([float(v) for v in ratio.split(",")])
            idx = idx[ratio > intersection_threshold]
            ratio = ratio[ratio > intersection_threshold]
            for i in range(len(idx)):
                filenames.append(key)
                data.append((idx[i], ratio[i]))
        data = np.array(data)
        if len(filenames) > patches_count_max:
            selected = np.random.choice(
                len(filenames),
                patches_count_max,
                replace=False,
                p=data[:, 1] / data[:, 1].sum(),
            )
            data = data[selected, :]
            filenames = [filenames[i] for i in selected]
        return [filenames, data]

    blocks_df["clean_patches_relation"] = blocks_df.patches_relation.apply(
        clean_patches_relation
    )
    blocks_df["patches_count"] = blocks_df.clean_patches_relation.apply(
        lambda x: x[1].shape[0]
    )
    blocks_df = blocks_df[blocks_df.patches_count > 0]
    blocks_df = blocks_df.reset_index(drop=True)
    return blocks_df


def get_unique_patches(blocks_df):
    patches_blocks = {}
    for i, row in blocks_df.iterrows():
        relation_list = row.clean_patches_relation[0]
        idx = row.clean_patches_relation[1][:, 0]
        files = [f"{relation_list[j]}_{int(idx[j])}" for j in range(len(idx))]
        for file in files:
            if file in patches_blocks.keys():
                patches_blocks[file].append(i)
            else:
                patches_blocks[file] = [i]
    return patches_blocks


def get_model(k):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.AutoEncoderResnetExtractor(dims=[2048, 1024, 256, 128])
    model_dec = models.DEC(n_clusters=k, embedding_dim=128, encoder=model.encoder)
    model_dec.load_state_dict(torch.load(f"../models/.../model.pt"))
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


def fraction_of_patches_cluster(blocks_df, k, filenames, clusters, patches_blocks):
    x = np.zeros((blocks_df.shape[0], k))
    for i, (file, cluster) in enumerate(zip(filenames, clusters)):
        for b in patches_blocks[file]:
            x[b, cluster] += 1
    x_sum = x.sum(axis=1)
    x = x / x_sum[:, None]
    x = pd.DataFrame(x, columns=[f"cluster_{i}" for i in range(k)])
    x = x.loc[:, x.sum(axis=0) > 0]
    x["count"] = x_sum
    return x


def distances_of_patches_cluster(
    blocks_df, k, filenames, clusters_distances, patches_blocks
):
    x = np.zeros((blocks_df.shape[0], k))
    for i, (file, distances) in enumerate(zip(filenames, clusters_distances)):
        for b in patches_blocks[file]:
            x[b, :] += distances
    x_sum = x.sum(axis=1)
    x = x / x_sum[:, None]
    x = pd.DataFrame(x, columns=[f"cluster_{i}" for i in range(k)])
    x = x.loc[:, x.sum(axis=0) > 0]
    x["count"] = x_sum
    return x


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


if __name__ == "__main__":
    ...
