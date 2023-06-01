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
    def __init__(self, filenames, resize=None, resnet=False):
        self.filenames = filenames
        self.preprocess = [transforms.ToTensor()]
        if resize is not None:
            self.preprocess.append(transforms.Resize(resize))
        if resnet:
            self.preprocess.append(
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                )
            )
        self.preprocess = transforms.Compose(self.preprocess)

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        img = Image.open(self.filenames[idx])
        img = self.preprocess(img)
        return img, self.filenames[idx]


class SmallPatchesDataset(torch.utils.data.Dataset):
    def __init__(self, filenames, resnet=False):
        self.filenames = filenames
        self.preprocess = [transforms.ToTensor()]
        if resnet:
            self.preprocess.append(
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                )
            )
        self.preprocess = transforms.Compose(self.preprocess)

    def __len__(self):
        return int(len(self.filenames) * 4)

    def __getitem__(self, idx):
        i = idx // 4
        j = idx % 4
        row = j // 2
        column = j % 2
        img = Image.open(self.filenames[i])
        img = img.crop((column * 112, row * 112, (column + 1) * 112, (row + 1) * 112))
        img = self.preprocess(img)
        return img, self.filenames[i]


def save_samples_patch(output_dir="patches", size=224):
    import rasterio

    ### loading the tifs
    sample_scenes = gpd.read_file("../data/output/downloaded_scenes_metadata.geojson")
    filenames = []
    im = Image.new("RGB", (size, size), "black")
    for i, row in tqdm(sample_scenes.iterrows()):
        tif = rasterio.open(f"../data/output/unzipped_files/{row.tif_filename}")
        row = gpd.GeoDataFrame(pd.DataFrame(row).T)
        patches_df = []
        patches, patches_rects = separate_tif_into_patches(tif, row, False, size=size)
        filename = row.tif_filename.values[0].replace(".tif", "")
        entity_id = filename.split("_")[0]

        j = 0
        while len(patches) > 0:
            patch = patches.pop(0)
            patches_rect = patches_rects.pop()
            im.paste(Image.fromarray(patch), (0, 0))
            im.save(f"../data/output/{output_dir}/{filename}_{j}.png")
            filenames.append(f"../data/output/{output_dir}/{filename}_{j}.png")
            patches_df.append([entity_id, j, f"{filename}_{j}.png", patches_rect])
            j += 1

        patches_df = gpd.GeoDataFrame(
            patches_df,
            columns=["entity_id", "patche_id", "patche_filename", "geometry"],
        )
        patches_df.to_file(f"../data/output/{output_dir}/{filename}.geojson")

    return filenames


def get_sample_patches_dataset(filenames=None, resize=None, resnet=False):
    if filenames is None:
        filenames = save_samples_patch()
    dataset = PatchesDataset(filenames, resize=resize, resnet=resnet)
    return dataset


def get_filenames():
    filenames = os.listdir("../data/output/patches")
    filenames = [
        os.path.join("../data/output/patches", f) for f in filenames if f[-3:] == "png"
    ]
    np.random.shuffle(filenames)
    return filenames


if __name__ == "__main__":
    dataset = get_sample_patches_dataset()
    print(len(dataset))
    print(dataset[0][0].shape)
