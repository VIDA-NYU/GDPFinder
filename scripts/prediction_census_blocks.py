import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from tqdm import tqdm

import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

import models
import data
import utils


def cluster_patches(blocks_df, model):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    filenames = [f for l in blocks_df.filenames.tolist() for f in l]
    dataset = data.SmallPatchesDataset(filenames)
    dl = DataLoader(dataset, batch_size=500, shuffle=False)
    clusters = utils.get_clusters(dl, model)
    clusters_distance = utils.get_clusters_distances(dl, model)
    n_files = blocks_df.filenames.apply(len).tolist()
    clusters = np.split(clusters, np.cumsum(n_files)[:-1])
    clusters_distance = np.split(clusters_distance, np.cumsum(n_files)[:-1])

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
        "n_estimators": [25, 50, 100],
        "max_depth": [10, 20, 50],
        "min_samples_leaf": [1, 5, 10],
    }
    clf = GridSearchCV(rf, parameters, n_jobs=-1, cv = 3)
    clf.fit(x_train, y_train)
    r2_train, r2_test, mae_train, mae_test = eval(clf, x_train, y_train, x_test, y_test)
    return r2_train, r2_test, mae_train, mae_test, clf


def eval_model(blocks_train, blocks_val, blocks_test, k):   
    results = []
    for method in ["fraction"]:#, "distance"]: #testing not using distance TEMPORARY
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

        for target in tqdm(["mhi", "density", "ed_attain"]):
            y_train = blocks_train[target].values
            y_val = blocks_val[target].values
            y_test = blocks_test[target].values
            r2_train, r2_val, mae_train, mae_val, clf = grid_search_rf(
                x_train, y_train, x_val, y_val
            )
            r2_test, _, mae_test, _ = eval(clf, x_test, y_test, x_test, y_test)
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
                    "best_params": clf.best_params_
                }
            )
    return pd.DataFrame(results)
