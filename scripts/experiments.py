import os
import torch
import numpy as np
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
    # train.train_reconstruction_feature_extraction(
    #     model,
    #     dl_train,
    #     dl_val,
    #     epochs=5,
    #     dir=f"../models/AE_extractor_resnet50_{str (dims)}/",
    # )
    model.load_state_dict(torch.load(f"../models/AE_extractor_resnet50_{str(dims)}/model.pt"))
    model.eval()
    for dl_name, dl in [("train", dl_train), ("val", dl_val), ("test", dl_test)]:
        embeddings = utils.get_embeddings(dl, model.encoder, device)
        np.save(f"../models/AE_extractor_resnet50_{str(dims)}/embeddings_{dl_name}.npy", embeddings)


def resnet_extractor_experiment(dims, n_clusters=100, train_ae = True, train_dec = True):
    dl_train, dl_val, _ = data.generate_datasets(
        patches_count_max=50, batch_size=256
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.AutoEncoderResnetExtractor(
        dims=dims, denoising=True
    ).to(device)
    if train_ae:
        embeddings = train.train_reconstruction_feature_extraction(
            model,
            dl_train,
            dl_val,
            epochs=5,
            return_embeddings=True,
            dir=f"../models/AE_extractor_resnet50_{str (dims)}/",
        )
    else:
        model.load_state_dict(
            torch.load(f"../models/AE_extractor_resnet50_{str(dims)}/model.pt")
        )
        model.eval()
        embeddings = utils.get_embeddings(dl_train, model.encoder, device)
    
    if train_dec:
        kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init=10).fit(embeddings)
        clusters = kmeans.labels_
        joblib.dump(kmeans, f"../models/AE_extractor_resnet50_{str(dims)}/kmeans_{n_clusters}_clusters.pkl")
        utils.plot_cluster_results(
            dl_train.dataset,
            clusters,
            dir=f"../models/AE_extractor_resnet50_{str(dims)}/kmeans_{n_clusters}_clusters",
        )
        centers = torch.tensor(kmeans.cluster_centers_)
        model_dec = models.DEC(
            n_clusters=n_clusters,
            embedding_dim=dims[-1],
            encoder=model.encoder,
            cluster_centers=centers,
        ).to(device)
        results = prediction_census_blocks.eval_model(model_dec, n_clusters)
        results.to_csv(f"../models/AE_extractor_resnet50_{str(dims)}/results_{n_clusters}_clusters.csv")

        clusters = train.train_clustering(
            model_dec,
            dl_train,
            dl_val,
            epochs=5,
            dir=f"../models/DEC_extractor_resnet50_{str(dims)}_{n_clusters}/",
        )
        utils.plot_cluster_results(
            dl_train.dataset,
            clusters,
            dir=f"../models/DEC_extractor_resnet50_{str(dims)}_{n_clusters}/dec_{n_clusters}_clusters",
        )
        results = prediction_census_blocks.eval_model(model_dec, n_clusters)
        results.to_csv(f"../models/DEC_extractor_resnet50_{str(dims)}_{n_clusters}/results_{n_clusters}_clusters.csv")
    else:
        kmeans = joblib.load(f"../models/AE_extractor_resnet50_{str(dims)}/kmeans_{n_clusters}_clusters.pkl")
        centers = torch.tensor(kmeans.cluster_centers_)
        model_dec = models.DEC(
            n_clusters=n_clusters,
            embedding_dim=dims[-1],
            encoder=model.encoder,
            cluster_centers=centers,
        ).to(device)
        results = prediction_census_blocks.eval_model(model_dec, n_clusters)
        results.to_csv(f"../models/AE_extractor_resnet50_{str(dims)}/results_{n_clusters}_clusters.csv")
        model_dec.load_state_dict(
            torch.load(f"../models/DEC_extractor_resnet50_{str(dims)}_{n_clusters}/model.pt")
        )
        model.to(device)
        results = prediction_census_blocks.eval_model(model_dec, n_clusters)
        results.to_csv(f"../models/DEC_extractor_resnet50_{str(dims)}_{n_clusters}/results_{n_clusters}_clusters.csv")

   

if __name__ == "__main__":
    np.random.seed(42)
    train_auto_encoder_extractor([2048, 512, 128, 64])
    # resnet_extractor_experiment([2048, 512, 128, 64], 100, False, True)
    # resnet_extractor_experiment([2048, 512, 128, 64], 50, False, True)
    # resnet_extractor_experiment([2048, 512, 128, 64], 20, False, True)
    train_auto_encoder_extractor([2048, 512, 128, 32])
    train_auto_encoder_extractor([2048, 512, 128, 10])
    
