import numpy as np
import pandas as pd
from tqdm import tqdm
from ast import literal_eval
from PIL import Image
import torch
from torchvision import transforms


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


def generate_dataloaders( 
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


def generate_dataframes(patches_count_max = 50):
    blocks_train = pd.read_csv(f"../data/blocks_patches_relation_train_{patches_count_max}.csv")
    blocks_val = pd.read_csv(f"../data/blocks_patches_relation_val_{patches_count_max}.csv")
    blocks_test = pd.read_csv(f"../data/blocks_patches_relation_test_{patches_count_max}.csv")
    blocks_train["filenames"] = blocks_train["filenames"].apply(literal_eval)
    blocks_val["filenames"] = blocks_val["filenames"].apply(literal_eval)
    blocks_test["filenames"] = blocks_test["filenames"].apply(literal_eval)
    return blocks_train, blocks_val, blocks_test