import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Polygon, MultiPolygon
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
import os


def create_files_df():
    """
    Function that will look into every tif image inside the folder data/output/unzipped_files
    and obtain the metadata of the scene that it is related based on the data/scenes_metadata folder.
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


def separate_tif_into_patches(tif, shp, mask_img=True, size=224):
    """
    Mask the tif image with the boundaries of the city by adding black pixels.
    After, crop it into patches with defined size and overlap.

    Inputs:
        tif: rasterio object (tif image)
        shp: geopandas dataframe with the metadata of the scene
        mask_img: boolean to mask the tif image
        size: size of the patches

    Outputs:
        patches: list of numpy arrays (patches)
    """
    from rasterio.mask import mask

    # get the boundaries of the scene city
    cities_shp = gpd.read_file("../data/CityBoundaries.shp").to_crs(tif.crs)
    cities_shp["city_name"] = cities_shp.NAME.apply(
        lambda x: x.lower().replace(" ", "_").replace("-", "_")
    )
    cities_shp["state_name"] = cities_shp.ST.apply(lambda x: x.lower())
    city = shp.city.values[0]
    state = shp.state.values[0]
    geo = cities_shp[
        (cities_shp.city_name == city) & (cities_shp.state_name == state)
    ].geometry.values[0]
    if type(geo) == Polygon:
        geo = [geo]
        geo = MultiPolygon(geo)

    if mask_img:
        out_image, _ = mask(tif, geo, filled=True)
    else:
        out_image = tif.read()

    # crop into patches
    patches = []
    patches_rects = []
    n_horizontal = out_image.shape[1] // size
    n_vertical = out_image.shape[2] // size
    lon_step = (shp.bounds.maxx.item() - shp.bounds.minx.item()) / n_horizontal
    lat_step = (shp.bounds.maxy.item() - shp.bounds.miny.item()) / n_vertical
    lon_start = shp.bounds.minx.item()
    lat_start = shp.bounds.miny.item()
    for i in range(n_horizontal):
        for j in range(n_vertical):
            i1 = i * size
            i2 = i1 + size
            j1 = j * size
            j2 = j1 + size
            lon1 = i * lon_step + lon_start
            lon2 = lon1 + lon_step
            lat1 = j * lat_step + lat_start
            lat2 = lat1 + lat_step
            patches.append(out_image[:3, i1:i2, j1:j2].transpose(1, 2, 0))
            patches_rects.append(
                Polygon([[lon1, lat1], [lon2, lat1], [lon2, lat2], [lon1, lat2]])
            )
            if np.sum(patches[-1]) == 0:
                patches.pop()
                patches_rects.pop()

    return patches, patches_rects


def plot_tif_image(filename, save_path=None):
    """
    Plot the tif image from filename with a really lower resolution.
    (It can change the picture appearence when reducing resolution)

    Inputs:
        filename: string with the name of the tif file
        save_path: string with the path to save the image, if None, it will show the image
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
    # )
    # separate_tif_into_patches(tif, test_sample)
