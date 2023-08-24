import os
import torch
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import joblib

import data
import models
import train
import utils
import prediction_census_blocks 


def train_auto_encoder_extractor(dims):
    dl_train, dl_val, dl_test = data.generate_dataloaders(
        patches_count_max=50, batch_size=256
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.AutoEncoderResnetExtractor(
        dims=dims, denoising=True
    ).to(device)
    train.train_reconstruction_feature_extraction(
        model,
        dl_train,
        dl_val,
        epochs=5,
        dir=f"../models/AE_extractor_resnet50_{str (dims)}/",
    )
    model.eval()
    for dl_name, dl in [("train", dl_train), ("val", dl_val), ("test", dl_test)]:
        embeddings = utils.get_embeddings(dl, model.encoder, device)
        np.save(f"../models/AE_extractor_resnet50_{str(dims)}/embeddings_{dl_name}.npy", embeddings)


def train_kmeans_dec(dims, n_clusters):
    dl_train, dl_val, dl_test = data.generate_dataloaders(
        patches_count_max=50, batch_size=256
    )
    blocks_train, blocks_val, blocks_test = data.generate_dataframes()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.AutoEncoderResnetExtractor(
        dims=dims, denoising=True
    ).to(device)
    model.load_state_dict(torch.load(f"../models/AE_extractor_resnet50_{str(dims)}/model.pt"))
    embeddings = np.load(f"../models/AE_extractor_resnet50_{str(dims)}/embeddings_train.npy")

    dir=f"../models/DEC_extractor_resnet50_{str(dims)}_{n_clusters}/"
    os.makedirs(dir, exist_ok=True)

    # kmeans
    kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init=10).fit(embeddings)
    joblib.dump(kmeans, f"../models/DEC_extractor_resnet50_{str(dims)}_{n_clusters}/kmeans_{n_clusters}_clusters.pkl")
    model_dec = models.DEC(
        n_clusters=n_clusters,
        embedding_dim=dims[-1],
        encoder=model.encoder,
        cluster_centers=torch.tensor(kmeans.cluster_centers_),
    ).to(device)
    model_dec.eval()
    for ds_name, df in [("train", blocks_train), ("val", blocks_val), ("test", blocks_test)]:
        df = prediction_census_blocks.cluster_patches(df, model_dec)
        df.to_pickle(f"../models/DEC_extractor_resnet50_{str(dims)}_{n_clusters}/blocks_{ds_name}_kmeans_{n_clusters}.pkl")

    # dec
    model_dec.train()
    train.train_clustering(
        model_dec,
        dl_train,
        dl_val,
        epochs=5,
        dir=f"../models/DEC_extractor_resnet50_{str(dims)}_{n_clusters}/",
    )
    model_dec.eval()
    for ds_name, df in [("train", blocks_train), ("val", blocks_val), ("test", blocks_test)]:
        df = prediction_census_blocks.cluster_patches(df, model_dec)
        df.to_pickle(f"../models/DEC_extractor_resnet50_{str(dims)}_{n_clusters}/blocks_{ds_name}_dec_{n_clusters}.pkl")

def train_regression(dims, k):
    # loading census blocks
    census = pd.read_csv("../data/crop_filenames.csv")
    census["data"] = census["Filename"].apply(lambda x : x.split(".ti")[0].split("_"))
    census = census["data"].apply(pd.Series)
    census.columns = ['state', 'county', 'tract', 'block group', "density", "mhi", "ed_attain"]
    for col in census.columns:
        census[col] = census[col].astype(float)
    for col in ['state', 'county', 'tract', 'block group']:
        census[col] = census[col].astype(int)

    # with kmeans
    blocks_train = pd.read_pickle(f"../models/DEC_extractor_resnet50_{dims}_{k}/blocks_train_kmeans_{k}.pkl")
    blocks_val = pd.read_pickle(f"../models/DEC_extractor_resnet50_{dims}_{k}/blocks_val_kmeans_{k}.pkl")
    blocks_test = pd.read_pickle(f"../models/DEC_extractor_resnet50_{dims}_{k}/blocks_test_kmeans_{k}.pkl")
    for d in [blocks_train, blocks_val, blocks_test]:
        d = d.drop(columns = ["mhi", "area", "year", "ed_attain", "density"])
        d = pd.merge(d, census, on = ['state', 'county', 'tract', 'block group'])
    results = prediction_census_blocks.eval_model(blocks_train, blocks_val, blocks_test, k)
    results.to_csv(f"../models/DEC_extractor_resnet50_{dims}_{k}/result_regression_kmeans_{k}.csv")

    # with dec
    blocks_train = pd.read_pickle(f"../models/DEC_extractor_resnet50_{dims}_{k}/blocks_train_dec_{k}.pkl")
    blocks_val = pd.read_pickle(f"../models/DEC_extractor_resnet50_{dims}_{k}/blocks_val_dec_{k}.pkl")
    blocks_test = pd.read_pickle(f"../models/DEC_extractor_resnet50_{dims}_{k}/blocks_test_dec_{k}.pkl")
    for d in [blocks_train, blocks_val, blocks_test]:
        d = d.drop(columns = ["mhi", "area", "year", "ed_attain", "density"])
        d = pd.merge(d, census, on = ['state', 'county', 'tract', 'block group'])
    results = prediction_census_blocks.eval_model(blocks_train, blocks_val, blocks_test, k)
    results.to_csv(f"../models/DEC_extractor_resnet50_{dims}_{k}/result_regression_dec_{k}.csv")


if __name__ == "__main__":
    np.random.seed(42)
    # train_auto_encoder_extractor([2048, 512, 128, 64])
    # train_auto_encoder_extractor([2048, 512, 128, 32])
    # train_auto_encoder_extractor([2048, 512, 128, 10])

    #for latent_dim in [32]:#, 64, 10]: #already run with 32 and 64
    #    for k in [100, 50, 20]:
    #        train_kmeans_dec([2048, 512, 128, latent_dim], k)
    
