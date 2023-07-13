import numpy as np
import pandas as pd
import geopandas as gpd
from tqdm import tqdm
import os
from PIL import Image
import torch
from torchvision import transforms

from handle_tif_images import separate_tif_into_patches


class PatchesDataset(torch.utils.data.Dataset):
    def __init__(self, filenames, resize=None):
        self.filenames = filenames
        self.preprocess = [transforms.ToTensor()]
        if resize is not None:
            self.preprocess.append(transforms.Resize(resize))
        self.preprocess = transforms.Compose(self.preprocess)

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        img = Image.open(self.filenames[idx])
        img = self.preprocess(img)
        return img


class SmallPatchesDataset(torch.utils.data.Dataset):
    def __init__(self, filenames, resize=None):
        self.filenames, self.idx = list(zip(*[(f[:-2], int(f[-1])) for f in filenames]))
        self.preprocess = [transforms.ToTensor()]
        if resize is not None:
            self.preprocess.append(transforms.Resize(resize, antialias=True))
        self.preprocess = transforms.Compose(self.preprocess)

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, i):
        filename = self.filenames[i]
        j = self.idx[i]
        row = j // 2
        column = j % 2
        img = Image.open(filename)
        img = img.crop((column * 112, row * 112, (column + 1) * 112, (row + 1) * 112))
        img = self.preprocess(img)
        return img


def get_filenames(n):
    cities_folders = os.listdir("../data/output/patches")
    filenames = []
    for city_folder in cities_folders:
        patches_files = os.listdir("../data/output/patches/" + city_folder)
        patches_files = [
            f"../data/output/patches/{city_folder}/{p}"
            for p in patches_files
            if p.endswith(".png")
        ]
        if len(patches_files) > n:
            patches_files = np.random.choice(
                patches_files, size=n, replace=False
            ).tolist()
        filenames += patches_files

    np.random.shuffle(filenames)
    return filenames


def get_filenames_small_patches(n):
    cities_folders = os.listdir("../data/output/patches")
    filenames = []
    for city_folder in cities_folders:
        patches_files = os.listdir("../data/output/patches/" + city_folder)
        patches_files = [
            f"../data/output/patches/{city_folder}/{p}"
            for p in patches_files
            if p.endswith(".png")
        ]
        if len(patches_files) > n:
            patches_files = np.random.choice(
                patches_files, size=n, replace=False
            ).tolist()
        filenames += patches_files

    np.random.shuffle(filenames)
    filenames = np.repeat(filenames, 4)
    filenames = np.array([filenames[i] + f" {i % 4}" for i in range(len(filenames))])
    return filenames


def get_filenames_center_blocks(intersection_threshold=0.25, patches_count_max=50):
    blocks_df = gpd.read_file("../data/census_blocks_patches_v2.geojson")
    # cleaning blocks with missing data
    blocks_df = blocks_df[blocks_df.mhi > 0]
    blocks_df = blocks_df.dropna()
    blocks_df = blocks_df[blocks_df.patches_relation.apply(len) > 0]

    def clean_patches_relation(s):
        s = s.split("\n")
        s = dict([x.split(":") for x in s])
        filenames = []
        data = []
        for key, value in s.items():
            value = value.split(" ")
            idx = np.array([float(v) for v in value[0].split(",")])
            ratio = np.array([float(v) for v in value[1].split(",")])
            idx = idx[ratio > intersection_threshold]
            ratio = ratio[ratio > intersection_threshold]
            for i in range(len(idx)):
                data.append([idx[i], ratio[i]])
                filenames.append(key)
        data = np.array(data)
        if len(filenames) > patches_count_max:
            selected = np.random.choice(
                len(filenames),
                size=patches_count_max,
                replace=False,
                p=data[:, 1] / data[:, 1].sum(),
            )
            data = data[selected, :]
            filenames = [filenames[i] for i in selected]
        return [filenames, data]

    blocks_df["clean_patches_relation"] = blocks_df.patches_relation.apply(
        clean_patches_relation
    )
    blocks_df["n_patches"] = blocks_df["clean_patches_relation"].apply(
        lambda x: x[1].shape[0]
    )
    blocks_df = blocks_df[blocks_df.n_patches > 0]

    filenames = blocks_df.clean_patches_relation.apply(
        lambda x: [x[0][i] + f" {int(x[1][i, 0])}" for i in range(len(x[0]))]
    ).sum()
    np.random.shuffle(filenames)
    return filenames
