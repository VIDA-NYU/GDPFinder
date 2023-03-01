import pandas as pd
import geopandas as gpd
from shapely.geometry import Polygon
from tqdm import tqdm
import sys
from time import time
import logging

sys.path.append("m2m-api/")
from api import M2M


def connect_to_m2m_api():
    username = "giovanivaldrighi"  # add your credentials
    password = "@ Poplio14usg"  # add your credentials
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
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    msa_shp = gpd.read_file("../data/msa_pop_biggest.shp")
    msa_shp["area"] = msa_shp["geometry"].area
    msa_shp = msa_shp.sort_values("area")
    print("Smallest metropolitan area:", msa_shp.iloc[0]["NAME"])
    m2m = connect_to_m2m_api()
    # config search
    search_params = {
        "datasetName": "high_res_ortho",
        "geoJsonType": "Polygon",
        "geoJsonCoords": [list(msa_shp.iloc[0]["geometry"].exterior.coords)],
        "maxResults": 10,
    }

    download_dir = clean_dir_name(msa_shp.iloc[0]["NAME"])
    scenes = m2m.searchScenes(**search_params)
    start = time()
    downloadMetadata = m2m.retrieveScenes(
        "high_res_ortho", scenes, download_dir=download_dir
    )
    end = time()

    print(f"Downloaded {len(downloadMetadata)} scenes in {end - start:.2f} seconds")


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


def search_all_scenes_ny():
    """Download the lastest scens for New York"""
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    msa_shp = gpd.read_file("../data/CityBoundaries.shp")
    msa_shp = msa_shp[msa_shp.NAME == "New York"]
    msa_shp = msa_shp.to_crs("EPSG:4326")

    m2m = connect_to_m2m_api()
    scenes_results = []
    # search all the scenes from "high_res_ortho"
    for p in msa_shp.iloc[0]["geometry"].geoms:
        coords = [list(p.exterior.coords)]
        # config search
        search_params = {
            "datasetName": "high_res_ortho",
            "geoJsonType": "Polygon",
            "geoJsonCoords": coords,
            "maxResults": 49999.0,
        }
        scenes = m2m.searchScenes(**search_params)

        for r in scenes["results"]:
            scenes_results.append(
                {
                    "dataset_name": "high_res_ortho",
                    "entity_id": r["entityId"],
                    "spatial_coverage": r["spatialCoverage"]["coordinates"],
                    "start_date": r["temporalCoverage"]["startDate"],
                    "end_date": r["temporalCoverage"]["endDate"],
                }
            )

    # search all the scenes from "high_res_ortho"
    for p in msa_shp.iloc[0]["geometry"].geoms:
        coords = [list(p.exterior.coords)]
        # config search
        search_params = {
            "datasetName": "naip",
            "geoJsonType": "Polygon",
            "geoJsonCoords": coords,
            "maxResults": 49999,
        }
        scenes = m2m.searchScenes(**search_params)

        for r in scenes["results"]:
            scenes_results.append(
                {
                    "dataset_name": "naip",
                    "entity_id": r["entityId"],
                    "spatial_coverage": r["spatialCoverage"]["coordinates"],
                    "start_date": r["temporalCoverage"]["startDate"],
                    "end_date": r["temporalCoverage"]["endDate"],
                }
            )

    scenes_results = pd.DataFrame(scenes_results)
    # some cleaning on the dataframe and printing results
    scenes_results["geometry"] = scenes_results["spatial_coverage"].apply(
        lambda x: Polygon(x[0])
    )
    # convert start_date and end_date to datetime
    scenes_results["start_date"] = pd.to_datetime(
        scenes_results["start_date"].apply(lambda x: x[:10])
    )
    scenes_results["end_date"] = pd.to_datetime(
        scenes_results["end_date"].apply(lambda x: x[:10])
    )
    print("Total of scenes for New York:", len(scenes_results))
    print("Distribution between the two datasets:")
    print(scenes_results["dataset_name"].value_counts())
    print("Distribution between the years for high_res_ortho:")
    print(
        scenes_results[scenes_results["dataset_name"] == "high_res_ortho"][
            "start_date"
        ].dt.year.value_counts()
    )
    print("Distribution between the years for naip:")
    print(
        scenes_results[scenes_results["dataset_name"] == "naip"][
            "start_date"
        ].dt.year.value_counts()
    )

    # convert to geodataframe and save
    scenes_results = gpd.GeoDataFrame(
        scenes_results, geometry="geometry", crs="EPSG:4326"
    )
    scenes_results = scenes_results.drop(columns=["spatial_coverage"])
    # return date columns to string
    scenes_results["start_date"] = scenes_results["start_date"].astype(str)
    scenes_results["end_date"] = scenes_results["end_date"].astype(str)
    scenes_results.to_file("../data/ny_scenes.shp")

    return


def download_scenes_ny():
    scenes_results = gpd.read_file("../data/ny_scenes.shp")

    # filter only 2021 and naip
    scenes_results = scenes_results[
        (scenes_results["dataset_na"] == "naip")
        & (scenes_results["start_date"].str.contains("2021"))
    ]
    scenes = {"results": []}
    for i, row in scenes_results.iterrows():
        scenes["results"].append({"entityId": row["entity_id"]})

    print("Total of scenes to download:", len(scenes["results"]))

    m2m = connect_to_m2m_api()
    download_dir = "./new_york"
    # download scenes
    m2m.retrieveScenes("naip", scenes, download_dir=download_dir)


if __name__ == "__main__":
    # download_data_test()
    # estimate_size()
    # search_all_scenes_ny()
    download_scenes_ny()
