import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import rasterio
from rasterio.mask import mask
from rasterio.plot import show
import matplotlib.pyplot as plt
from matplotlib import cm
from PIL import Image
import os
from sklearn.feature_extraction import image


def create_files_df():
    """
    Function that will look into every tif image inside the folder data/output/unzipped_files
    and obtain the metadata of the scene that it is related.
    """
    # creating dataframe of unzipped files
    unzipped_files = [
        f for f in os.listdir("../data/output/unzipped_files") if f[-4:] == ".tif"
    ]
    unzipped_entity_id = [f.split("_")[0] for f in unzipped_files]
    unzipped_files_df = pd.DataFrame(
        {"tif_filename": unzipped_files, "entity_id": unzipped_entity_id}
    )

    # creating dataframe of scenes metadata
    shapefiles = [
        f for f in os.listdir("../data/scenes_metadata") if f[-8:] == ".geojson"
    ]
    scenes_shp = []
    for f in shapefiles:
        scenes_shp.append(gpd.read_file(f"../data/scenes_metadata/{f}"))
        scene_name = f.replace("_last_scenes.geojson", "")
        city = scene_name[:-3]
        state = scene_name[-2:]
        scenes_shp[-1]["city"] = city
        scenes_shp[-1]["state"] = state
        scenes_shp[-1]["shapefile_filename"] = f
    scenes_shp = gpd.GeoDataFrame(pd.concat(scenes_shp))
    df = pd.merge(unzipped_files_df, scenes_shp, on="entity_id", how="left")
    return gpd.GeoDataFrame(df)


def separate_tif_into_patches(tif, shp, size=224, overlap=8, plot_patches = False):
    # get the boundaries of the scene city
    cities_shp = gpd.read_file("../data/CityBoundaries.shp").to_crs(tif.crs)
    cities_shp["city_name"] = cities_shp.NAME.apply(lambda x : x.lower().replace(" ", "_").replace("-", "_"))
    cities_shp["state_name"] = cities_shp.ST.apply(lambda x : x.lower())
    city = shp.city.values[0]
    state = shp.state.values[0]
    geo = cities_shp[(cities_shp.city_name == city) & (cities_shp.state_name == state)].geometry.values[0]
    
    # mask it
    out_image, _ = mask(tif, geo, filled=True)
    
    # crop into patches
    patches = []
    n_horizontal = out_image.shape[1] // (size - overlap)
    n_vertical = out_image.shape[2] // (size - overlap)
    for i in range(n_horizontal):
        for j in range(n_vertical):
            i1 = i * (size - overlap)
            i2 = i1 + size
            j1 = j * (size - overlap)
            j2 = j1 + size
            patches.append(out_image[:3, i1:i2, j1:j2].transpose(1, 2, 0))
            if np.sum(patches[-1]) == 0:
                patches.pop()
    
    if plot_patches:
        for i, img in enumerate(patches):
            plt.axis(False)
            plt.imshow(img, interpolation="nearest")
            plt.savefig(f"../figures/testing_{i}.png")
            plt.close()
    
    return patches


def plot_tif_image(filename, save_path=None):
    """
    Plot the tif image from filename with a really lower resolution. (It can have some artefacts due to resolution reduction)
    """
    im = Image.open("../data/output/unzipped_files/" + filename)
    width, height = im.size
    new_height = int(1080 / width * height)
    im = im.resize((1080, new_height), resample=Image.Resampling.BILINEAR)
    im = np.array(im)[:, :, :3]
    plt.imshow(im)
    plt.axis(False)
    plt.title(f"scene {filename.split('_')[0]}")
    if not save_path is None:
        plt.savefig(save_path)
    else:
        plt.show()


if __name__ == "__main__":
    # saving metadata from the download tif images
    df = create_files_df()
    df.to_file("../data/output/downloaded_scenes_metadata.geojson")
    
    # testing ploting a random image
    # filename = (
    #    gpd.read_file("../data/output/downloaded_scenes_metadata.geojson")
    #    .sample()
    #    .tif_filename.values[0]
    # )
    # plot_tif_image(filename, "../figures/tif_plot.png")

    # getting patches for the tile of manhattan
    # test_sample = gpd.read_file("../data/output/downloaded_scenes_metadata.geojson")
    # test_sample = test_sample[test_sample.geometry.contains(Point([-74.004162, 40.708060]))].head(1)

    # tif = rasterio.open(
    #    f"../data/output/unzipped_files/{test_sample.tif_filename.values[0]}"
    #)
    #separate_tif_into_patches(tif, test_sample)