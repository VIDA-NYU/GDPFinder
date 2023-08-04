import numpy as np
import pandas as pd
from tqdm import tqdm
import os
from ast import literal_eval
from PIL import Image
import torch
from torchvision import transforms


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
    blocks_df = pd.read_csv("../data/blocks_patches_relation.csv")
    blocks_df["mhi"] = blocks_df["mhi"].apply(lambda x: np.nan if x < 0 else x)
    blocks_df = blocks_df.dropna()
    blocks_df.patches_relation = blocks_df.patches_relation.apply(literal_eval)
    blocks_df["n_patches"] = blocks_df.patches_relation.apply(len)
    blocks_df = blocks_df[blocks_df.n_patches > 0]

    all_filenames = []
    for i, row in tqdm(blocks_df.iterrows(), total=len(blocks_df)):
        filenames = []
        data = []
        s = row.patches_relation
        for key, value in s.items():
            value = np.array(value)
            value = value[value[:, 1] > intersection_threshold, :]
            for i in range(len(value)):
                data.append(value[i, :])
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
        all_filenames.extend(
            [
                f"../data/output/patches/{filenames[i]} {int(data[i, 0])}"
                for i in range(len(filenames))
            ]
        )

    np.random.shuffle(all_filenames)
    return all_filenames

def generate_datasets( 
        patches_count_max=50,
        batch_size = 96,
    ):
    blocks_df_train = pd.read_csv(f"../data/blocks_patches_relation_train_{patches_count_max}.csv")
    blocks_df_val = pd.read_csv(f"../data/blocks_patches_relation_val_{patches_count_max}.csv")
    blocks_df_test = pd.read_csv(f"../data/blocks_patches_relation_test_{patches_count_max}.csv")
    blocks_df_train["filenames"] = blocks_df_train["filenames"].apply(literal_eval)
    blocks_df_val["filenames"] = blocks_df_val["filenames"].apply(literal_eval)
    blocks_df_test["filenames"] = blocks_df_test["filenames"].apply(literal_eval)
    filenames_train = [f for l in blocks_df_train.filenames.tolist() for f in l]
    filenames_val = [f for l in blocks_df_val.filenames.tolist() for f in l]
    filenames_test = [f for l in blocks_df_test.filenames.tolist() for f in l]
    
    dl_train = torch.utils.data.DataLoader(
        SmallPatchesDataset(filenames_train),
        batch_size=batch_size,
    )
    dl_val = torch.utils.data.DataLoader(
        SmallPatchesDataset(filenames_val),
        batch_size=batch_size,
    )
    dl_test = torch.utils.data.DataLoader(
        SmallPatchesDataset(filenames_test),
        batch_size=batch_size,
    )

    return dl_train, dl_val, dl_test