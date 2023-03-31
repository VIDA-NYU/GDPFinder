import os
import geopandas as gpd
import pandas as pd
import numpy as np
import torch
import rasterio
from handle_tif_images import separate_tif_into_patches

class PatchesDataset(torch.utils.data.Dataset):
    def __init__(self, imgs):
        self.data = imgs
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx, :], 0

def get_sample_patches_dataset():
    ### loading the tifs
    sample_scenes = gpd.read_file("../data/output/downloaded_scenes_metadata.geojson")
    patches = []
    for i, row in sample_scenes.iterrows():
        tif = rasterio.open(
            f"../data/output/unzipped_files/{row.tif_filename}"
        )
        row = pd.DataFrame(row).T
        patches.append(
            separate_tif_into_patches(tif, row, plot_patches=False)
        )
    
    patches = sum(patches, [])
    patches = torch.tensor(np.array(patches).transpose(0, 3, 1, 2).astype(np.float32))
    dataset = PatchesDataset(patches)
    return dataset