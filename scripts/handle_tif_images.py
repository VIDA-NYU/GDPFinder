import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Polygon, MultiPolygon
from rasterio.mask import mask
import matplotlib.pyplot as plt
from PIL import Image
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


<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> 53445937bf601a934a93718bf1185e6b6b13c446
=======
>>>>>>> 53445937bf601a934a93718bf1185e6b6b13c446
def separate_tif_into_patches(tif, shp, mask_img = True, size=224):
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
<<<<<<< HEAD
=======
def separate_tif_into_patches(tif, shp, size=224, overlap=8, plot_patches = False):
>>>>>>> 33d3171 (initial)
=======
>>>>>>> 53445937bf601a934a93718bf1185e6b6b13c446
    # get the boundaries of the scene city
    cities_shp = gpd.read_file("../data/CityBoundaries.shp").to_crs(tif.crs)
    cities_shp["city_name"] = cities_shp.NAME.apply(lambda x : x.lower().replace(" ", "_").replace("-", "_"))
    cities_shp["state_name"] = cities_shp.ST.apply(lambda x : x.lower())
    city = shp.city.values[0]
    state = shp.state.values[0]
    geo = cities_shp[(cities_shp.city_name == city) & (cities_shp.state_name == state)].geometry.values[0]
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> 53445937bf601a934a93718bf1185e6b6b13c446
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
<<<<<<< HEAD
<<<<<<< HEAD
=======
    
    # mask it
    out_image, _ = mask(tif, geo, filled=True)
    
    # crop into patches
    patches = []
    n_horizontal = out_image.shape[1] // (size - overlap)
    n_vertical = out_image.shape[2] // (size - overlap)
>>>>>>> 33d3171 (initial)
=======
>>>>>>> 53445937bf601a934a93718bf1185e6b6b13c446
=======
>>>>>>> 53445937bf601a934a93718bf1185e6b6b13c446
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
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> 53445937bf601a934a93718bf1185e6b6b13c446
=======
>>>>>>> 53445937bf601a934a93718bf1185e6b6b13c446
            patches_rects.append(Polygon([[lon1, lat1], [lon2, lat1], [lon2, lat2], [lon1, lat2]]))
            if np.sum(patches[-1]) == 0:
                patches.pop()
                patches_rects.pop()
<<<<<<< HEAD
<<<<<<< HEAD
=======
    
    if plot_patches:
        for i, img in enumerate(patches):
            plt.axis(False)
            plt.imshow(img, interpolation="nearest")
            plt.savefig(f"../figures/testing_{i}.png")
            plt.close()
    
    return patches
>>>>>>> 33d3171 (initial)
=======
>>>>>>> 53445937bf601a934a93718bf1185e6b6b13c446
    
=======
    
>>>>>>> 53445937bf601a934a93718bf1185e6b6b13c446
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

<<<<<<< HEAD
<<<<<<< HEAD
    # tif = rasterio.open(
    #    f"../data/output/unzipped_files/{test_sample.tif_filename.values[0]}"
    #)
    #separate_tif_into_patches(tif, test_sample)
=======
    tif = rasterio.open(
        f"../data/output/unzipped_files/{test_sample.tif_filename.values[0]}"
    )
    separate_tif_into_patches(tif, test_sample)
>>>>>>> 33d3171 (initial)
=======
    # tif = rasterio.open(
    #    f"../data/output/unzipped_files/{test_sample.tif_filename.values[0]}"
    #)
    #separate_tif_into_patches(tif, test_sample)
>>>>>>> 53445937bf601a934a93718bf1185e6b6b13c446
