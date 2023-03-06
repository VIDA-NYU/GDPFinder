import pandas as pd
import geopandas as gpd
from shapely.geometry import Polygon, MultiPolygon
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


def clean_name(name):
    download_dir = name
    download_dir = download_dir.replace(", ", "_")
    download_dir = download_dir.replace(" ", "_")
    download_dir = download_dir.replace("-", "_")
    download_dir = download_dir.lower()
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

    download_dir = "./" + clean_name(msa_shp.iloc[0]["NAME"])
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
        download_dir = "./" + clean_name(row["NAME"])
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


def search_scenes(city = "New York", dataset = "naip", years = list(range(2018, 2022))):
    """
    Function that search for all scenes of the "city" from the two datasets "high_res_ortho" or "naip"
    and that from a year of the list "years" and saves into a shapefile with the metadata.

    Inputs:
        city - string with the city name
        dataset - string with the dataset name to be used, ["high_res_ortho", "naip"]
        years - list of years to select scenes
    """
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    print(f"Searching scenes for {city}")

    clean_city_name = clean_name(city)
    cities_shp = gpd.read_file("../data/CityBoundaries.shp")
    
    assert dataset in ["naip", "high_res_ortho"]
    assert city in cities_shp.NAME.values
    
    cities_shp = cities_shp[cities_shp.NAME == city]
    cities_shp = cities_shp.to_crs("EPSG:4326")

    m2m = connect_to_m2m_api()
    scenes_results = []

    # transform polygon to multipolygon (if necessary)
    cities_shp["geometry"] = cities_shp["geometry"].apply(lambda x : MultiPolygon([x]) if type(x) == Polygon else x)

    for p in cities_shp.iloc[0]["geometry"].geoms:
        coords = [list(p.exterior.coords)]
        search_params = {
            "datasetName": dataset,
            "geoJsonType": "Polygon",
            "geoJsonCoords": coords,
            "maxResults": 49999.0,
        }
        scenes = m2m.searchScenes(**search_params)
        for r in scenes["results"]:
            scenes_results.append(
                {
                    "dataset": dataset,
                    "entity_id": r["entityId"],
                    "spatial_coverage": r["spatialCoverage"]["coordinates"],
                    "start_date": r["temporalCoverage"]["startDate"],
                    "end_date": r["temporalCoverage"]["endDate"],
                }
            )

    print(f"Total of {len(scenes_results)} scenes found")
    if len(scenes_results) == 0:
        return None
    
    # create dataframe and do some cleaning
    scenes_results = pd.DataFrame(scenes_results, columns = ["dataset", "entity_id", "spatial_coverage", "start_date", "end_date"])
    scenes_results["geometry"] = scenes_results["spatial_coverage"].apply(
        lambda x: Polygon(x[0])
    )
    scenes_results["start_date"] = scenes_results["start_date"].apply(lambda x: x[:10])
    scenes_results["end_date"] = scenes_results["end_date"].apply(lambda x: x[:10])
    scenes_results["year"] = scenes_results["start_date"].apply(lambda x : x[:4]).astype(int)
    scenes_results = scenes_results[scenes_results.year.isin(years)]
    print(f"Total of scenes for {city}: {len(scenes_results)}")

    # convert to geodataframe and save
    scenes_results = gpd.GeoDataFrame(
        scenes_results, geometry="geometry", crs="EPSG:4326"
    )
    scenes_results = scenes_results.drop(columns=["spatial_coverage", "year"])
    scenes_results.to_file(f"../data/shapefiles/{clean_city_name}_scenes.shp")

    return scenes_results


def download_scenes(city = "New York"):
    """
    Function that download previously search scenes for the "city".

    Inputs:
        city - string with the city name
    """
    clean_city_name = clean_name(city)
    scenes_results = gpd.read_file(f"../data/shapefiles/{clean_city_name}_scenes.shp")
    dataset = scenes_results.dataset.values[0]
    scenes = {"results": []}
    for i, row in scenes_results.iterrows():
        scenes["results"].append({"entityId": row["entity_id"]})
    print("Total of scenes to download:", len(scenes["results"]))

    m2m = connect_to_m2m_api()
    download_dir = f"./../data/output/{clean_city_name}"
    # download scenes
    m2m.retrieveScenes(dataset, scenes, download_dir=download_dir)


def iterate_over_biggest_cities(n_biggest = 50,  dataset = "naip", years = list(range(2018, 2022))):
    """
    Iterate over the n_biggest cities downloading their scenes from the "dataset" dataset for any of the year in "years".

    Inputs:
        n_biggest - number of cities to select
        dataset - string with the dataset name to be used, ["high_res_ortho", "naip"]
        years - list of years to select scenes
    """
    cities_shp = gpd.read_file("../data/CityBoundaries.shp")
    cities_shp["area"] = cities_shp.geometry.area
    cities_shp = cities_shp.sort_values(by = "area", ascending = False)
    cities_shp = cities_shp.head(n_biggest)
    for i, row in cities_shp.iterrows():
        scenes_results = search_scenes(row["NAME"], dataset, years)
        if not scenes_results is None:
            download_scenes(row["NAME"])


def iterate_over_selected_cities(selected_cities, dataset = "naip", years = list(range(2018, 2022))):
    """
    Iterate over a list of select cities downloading their scenes from the "dataset" dataset for any of the year in "years".

    Inputs:
        selected_cities - list of string with selected cities
        dataset - string with the dataset name to be used, ["high_res_ortho", "naip"]
        years - list of years to select scenes
    """
    for name in selected_cities:
        scenes_results = search_scenes(name, dataset, years)
        if not scenes_results is None:
            download_scenes(name)



if __name__ == "__main__":
    # download_data_test()
    # estimate_size()
    #search_scenes("Denver")
    #download_scenes("Denver")

    iterate_over_biggest_cities(n_biggest = 10)
    #iterate_over_selected_cities(["Houston"])