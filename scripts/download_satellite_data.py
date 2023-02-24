import geopandas as gpd
from tqdm import tqdm
import sys

sys.path.append("m2m-api/")
from api import M2M


def connect_to_m2m_api():
    username = # add your credentials
    password = # add your credentials
    m2m = M2M(username, password)
    return m2m


def clean_dir_name(name):
    download_dir = name
    download_dir = download_dir.replace(", ", "_")
    download_dir = download_dir.replace(" ", "_")
    download_dir = download_dir.replace("-", "_")
    download_dir = download_dir.lower()
    download_dir = "./" + download_dir
    return download_dir


def download_data_test():
    """Download some scenes from the smallest MSA"""
    msa_shp = gpd.read_file("../data/msa_pop_biggest.shp")
    msa_shp["area"] = msa_shp["geometry"].area
    msa_shp = msa_shp.sort_values("area")
    print("Smallest metropolitan area:", msa_shp.iloc[0]["NAME"])

    m2m = connect_to_m2m_api()
    # config search
    search_params = {
        "datasetName": "naip",
        "geoJsonType": "Polygon",
        "geoJsonCoords": [list(msa_shp.iloc[0]["geometry"].exterior.coords)],
        "maxResults": 3,
    }
    download_dir = clean_dir_name(msa_shp.iloc[0]["NAME"])
    scenes = m2m.searchScenes(**search_params)
    downloadMetadata = m2m.retrieveScenes("naip", scenes, download_dir=download_dir)


def download_data():
    """Download scenes from all MSA"""
    msa_shp = gpd.read_file("../data/msa_pop_biggest.shp")
    m2m = connect_to_m2m_api()

    for i, row in tqdm(msa_shp.iterrows()):
        print(f"Download data from {row['NAME']}")
        # config search
        search_params = {
            "datasetName": "naip",
            "geoJsonType": "Polygon",
            "geoJsonCoords": [list(row["geometry"].exterior.coords)],
            "maxResults": 20,
        }
        download_dir = clean_dir_name(row["NAME"])
        scenes = m2m.searchScenes(**search_params)
        downloadMetadata = m2m.retrieveScenes("naip", scenes, download_dir=download_dir)


def estimate_size():
    """Count the number of scenes and estimate the storage size"""
    msa_shp = gpd.read_file("../data/msa_pop_biggest.shp")
    m2m = connect_to_m2m_api()
    total_scenes = 0
    for i, row in msa_shp.iterrows():
        try:
            # config search
            search_params = {
                "datasetName": "naip",
                "geoJsonType": "Polygon",
                "geoJsonCoords": [list(row["geometry"].exterior.coords)],
                "maxResults": 3,
            }
            scenes = m2m.searchScenes(**search_params)
            total_scenes += scenes["totalHits"]
            print(f"MSA {row['NAME']} has {scenes['totalHits']} scenes")
        except:
            print(f"Error with MSA {row['NAME']}")

    mb_by_scene = 400
    print(f"Total scenes: {total_scenes}")
    print(f"Total size: {(total_scenes * mb_by_scene)/ 1e6} TB")


if __name__ == "__main__":
    download_data_test()
    # estimate_size()
