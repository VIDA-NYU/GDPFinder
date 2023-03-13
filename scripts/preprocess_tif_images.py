import pandas as pd
import geopandas as gpd
import os



def create_files_df():
    """
    Function that will look into every tif image inside the folder data/output/unzipped_files
    and obtain the metadata of the scene that it is related.
    """
    # creating dataframe of unzipped files
    unzipped_files = [f for f in os.listdir("../data/output/unzipped_files") if f[-4:] == ".tif"]
    unzipped_entity_id = [f.split("_")[0] for f in unzipped_files]
    unzipped_files_df = pd.DataFrame({"tif_filename" : unzipped_files, "entity_id" : unzipped_entity_id})
    
    # creating dataframe of scenes metadata
    shapefiles = [f for f in os.listdir("../data/scenes_metadata") if f[-8:] == ".geojson"]
    scenes_shp = []
    for f in shapefiles:
        scenes_shp.append(gpd.read_file(f"../data/scenes_metadata/{f}"))
        scene_name = f.replace("_last_scenes.geojson", "")
        city = scene_name[:-3]
        state = scene_name[-2:]
        scenes_shp[-1]["city"] = city
        scenes_shp[-1]["state"] = state
        scenes_shp[-1]["shapefile_filename"] = f
    scenes_shp = pd.concat(scenes_shp)
    df = pd.merge(unzipped_files_df, scenes_shp, on = "entity_id", how="left")
    return df


if __name__ == "__main__":
    df = create_files_df()
    df.to_csv("../data/output/downloaded_scenes_metadata.csv", index = False)