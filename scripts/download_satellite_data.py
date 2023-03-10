import numpy as np
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


def get_clean_name(name):
    download_dir = name
    download_dir = download_dir.replace(", ", "_")
    download_dir = download_dir.replace(" ", "_")
    download_dir = download_dir.replace("-", "_")
    download_dir = download_dir.lower()
    return download_dir


def search_scenes(city = "New York", state = "NY", dataset = "naip", years = list(range(2018, 2022))):
    """
    Function that search for all scenes of the "city" from the two datasets "high_res_ortho" or "naip"
    and that from a year of the list "years" and saves into a shapefile with the metadata.

    Inputs:
        city - string with the city name
        state - string with the state name
        dataset - string with the dataset name to be used, ["high_res_ortho", "naip"]
        years - list of years to select scenes
    """

    print(f"Searching scenes for {city} - {state}")
    cities_shp = gpd.read_file("../data/CityBoundaries.shp")
    
    assert dataset in ["naip", "high_res_ortho"]
    assert city in cities_shp.NAME.values
    assert state in cities_shp.ST.values
    
    cities_shp = cities_shp[((cities_shp.NAME == city) & (cities_shp.ST == state))]

    assert cities_shp.shape[0] > 0

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

    print(f"Total of scenes found: {len(scenes_results)}")
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

    if len(scenes_results) == 0:
        return None

    # filtering scenes based on download options
    download_options = pd.DataFrame(m2m.downloadOptions(dataset, list(scenes_results.entity_id.values)))
    download_options = download_options[download_options.available]
    download_options = download_options[download_options.downloadSystem.isin(["dds", "dds_zip"])]
    download_options = download_options[download_options.productName == "Full Resolution"]
    download_options = download_options.drop_duplicates("entityId")

    scenes_results = scenes_results[scenes_results.entity_id.isin(download_options.entityId.values)]
    scenes_results = scenes_results.drop_duplicates(["entity_id"])
    print(f"Total scenes for the selected years: {scenes_results.shape[0]} ")
    # convert to geodataframe and save
    scenes_results = gpd.GeoDataFrame(
        scenes_results, geometry="geometry", crs="EPSG:4326"
    )
    scenes_results = scenes_results.drop(columns=["spatial_coverage", "year"])

    return scenes_results


def download_scenes(download_dir, scenes_results):
    """
    Function that download previously searched scenes and download into a directory.

    Inputs:
        download_dir - string with the download dir
        scenes_results - geopandas dataframe with scenes information
    """
    dataset = scenes_results.dataset.values[0]
    scenes = {"results": []}
    for i, row in scenes_results.iterrows():
        scenes["results"].append({"entityId": row["entity_id"]})
    print("Total of scenes to download:", len(scenes["results"]))

    m2m = connect_to_m2m_api()
    download_dir = f"./../data/output/{clean_city_name}_{clean_state_name}"
    # download scenes
    filter_options = {
        "downloadSystem": lambda x: x == "dds_zip" or x == "dds",
        "available": lambda x: x,
        "productName" : lambda x : x == "Full Resolution"
    }
    m2m.retrieveScenes(dataset, scenes, filterOptions = filter_options, download_dir = download_dir)


def iterate_over_selected_cities(selected_cities, dataset = "naip", years = list(range(2018, 2022))):
    """
    Iterate over a list of select cities downloading their scenes from the "dataset" dataset for any of the year in "years".

    Inputs:
        selected_cities - list of string with selected cities
        dataset - string with the dataset name to be used, ["high_res_ortho", "naip"]
        years - list of years to select scenes
    """
    for name in selected_cities:
        city, state = name.split("-")
        clean_city_name = get_clean_name(city)
        clean_state_name = get_clean_name(state)
        scenes_results = search_scenes(name, dataset, years)
        if not scenes_results is None:
            download_scenes(name, scenes_results)


def search_scenes_selected_cities_most_recent(selected_cities, dataset = "naip"):
    cities_shp = gpd.read_file("../data/CityBoundaries.shp")
    cities_shp = cities_shp.to_crs("EPSG:4326")
    for name in selected_cities:
        city, state = name.split("-")
        scenes_results = search_scenes(city, state, dataset, list(range(2000, 2022)))
        if scenes_results is None:
            continue
        scenes_results["year"] = scenes_results.start_date.apply(lambda x : int(x[:4]))
        year_values = list(scenes_results.year.unique())
        year_values.sort()
        year_values.reverse()
        filtered_cities_shp = cities_shp[(cities_shp.NAME == city) & (cities_shp.ST == state)]
        for year in year_values:
            scenes_results_year = scenes_results[scenes_results.year == year]
            if scenes_results_year.shape[0] == 0:
                continue
            if verify_scenes_covering(filtered_cities_shp, scenes_results_year):
                clean_city_name = get_clean_name(city)
                clean_state_name = get_clean_name(state)
                print(f"Scenes for {name} found for year {year}: {scenes_results_year.shape[0]}")
                scenes_results_year.to_file(f"../data/scenes_metadata/{clean_city_name}_{clean_state_name}_last_scenes.geojson")
                break

def download_scenes_selected_cities_most_recent(selected_cities, dataset = "naip"):
    for name in selected_cities:
        city, state = name.split("-")
        clean_city_name = get_clean_name(city)
        clean_state_name = get_clean_name(state)
        scenes_results = gpd.read_file(f"../data/scenes_metadata/{clean_city_name}_{clean_state_name}_last_scenes.geojson")
        download_scenes(city, state, scenes_results)

def get_biggest_gdp(n_biggest = 25):
    """Order the MSA based on the GDP values, and for the n_biggest biggest MSA, select the city with biggest area """
    gdp_df = pd.read_csv("../data/MSA_GDP.csv")[["GeoName", "2021"]]
    # remove "United States" line
    gdp_df = gdp_df[gdp_df.GeoName.apply(lambda x : x.find("United States") != 0)]

    gdp_df["GeoName"] = gdp_df.GeoName.apply(lambda x : x[:x.find(' (')])
    gdp_df["city"] = gdp_df.GeoName.apply(lambda x : x[:x.find(',')].strip().split("-"))
    gdp_df["state"] = gdp_df.GeoName.apply(lambda x : x[x.find(',') + 1:].strip().split("-"))
    gdp_df = gdp_df.sort_values("2021", ascending = False).head(n_biggest)

    cities_shp = gpd.read_file("../data/CityBoundaries.shp")
    cities_shp["area"] = cities_shp.geometry.area
    cities_shp = cities_shp.sort_values(by = "area", ascending = False)

    # create a new dataframe separating the cities of each MSA
    gdp_df_separated = []
    msa_i = 0
    for i, row in gdp_df.iterrows():
        for city in row["city"]:
            for state in row["state"]:
                filtered_shp = cities_shp[((cities_shp.NAME == city) & (cities_shp.ST == state))]
                if  filtered_shp.shape[0] > 0:
                    area = filtered_shp.geometry.values[0].area
                    gdp_df_separated.append([city, state, area, msa_i])
        msa_i += 1

    gdp_df_separated = pd.DataFrame(gdp_df_separated, columns = ["city", "state", "area", "msa_i"])

    # keep only the city of the biggest area for each MSA
    def keep_biggest(df):
        df = df[df.area == df.area.max()]
        return df
    gdp_df_separated = gdp_df_separated.groupby(["msa_i"]).apply(keep_biggest)
    biggest_cities = list((gdp_df_separated.city + "-" + gdp_df_separated.state).values)
    return biggest_cities


def verify_scenes_covering(city_shp, scenes_shp):
    """Verifies if the spatial extent of the scenes cover more than 70% of the area of the city"""
    scenes_total_cover = scenes_shp.to_crs("EPSG:32633").geometry.unary_union
    cities_total_cover = city_shp.to_crs("EPSG:32633").geometry.unary_union
    cities_total_cover = cities_total_cover.buffer(0) # little fix
    city_intersection = cities_total_cover.intersection(scenes_total_cover)
    intersection_ratio = city_intersection.area / cities_total_cover.area
    return intersection_ratio > 0.85


if __name__ == "__main__":

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    selected_cities = get_biggest_gdp(50)
    print(selected_cities)
    #search_scenes_selected_cities_most_recent(selected_cities)
    download_scenes_selected_cities_most_recent(selected_cities[:10])
