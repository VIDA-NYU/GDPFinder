import os
import geopandas as gpd
import pandas as pd
import numpy as np
import torch
from torchvision import transforms
import rasterio
from PIL import Image
from tqdm import tqdm
from handle_tif_images import separate_tif_into_patches


class PatchesDataset(torch.utils.data.Dataset):
    def __init__(self, filenames):
        self.filenames = filenames
        self.preprocess = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        # read and preprocess the image
        img = Image.open(self.filenames[idx])
        img = self.preprocess(img)
        return img, self.filenames[idx]


def save_samples_patch():
    ### loading the tifs
    sample_scenes = gpd.read_file("../data/output/downloaded_scenes_metadata.geojson")
    filenames = []

    for i, row in sample_scenes.iterrows():
        tif = rasterio.open(f"../data/output/unzipped_files/{row.tif_filename}")
        row = pd.DataFrame(row).T
        patches = separate_tif_into_patches(tif, row)
        filename = row.tif_filename.values[0].replace(".tif", "")

        for j, patch in enumerate(patches):
            im = Image.fromarray(patch)
            im.save(f"../data/output/patches/{filename}_{j}.png")
            filenames.append(f"../data/output/patches/{filename}_{j}.png")

    return filenames


def get_sample_patches_dataset(filenames=None):
    if filenames is None:
        filenames = save_samples_patch()
    dataset = PatchesDataset(filenames)
    return dataset
