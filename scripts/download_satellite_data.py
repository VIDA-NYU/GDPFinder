import numpy as np
import pandas as pd
import geopandas as gpd
from tqdm import tqdm
from shapely.geometry import Polygon, MultiPolygon
from zipfile import ZipFile
import sys
import os
import logging

sys.path.append("m2m-api/")
from api import M2M


def connect_to_m2m_api():
    username = "giovanivaldrighi"  # add your credentials
    password = "@ Poplio14usg"  # add your credentials
    m2m = M2M(username, password)
    return m2m


def get_clean_name(name):
    """
    Clean the strings with the name of the cities to use on the filenames.

    Inputs:
        name - string
    """
    download_dir = name
    download_dir = download_dir.replace(", ", "_")
    download_dir = download_dir.replace(" ", "_")
    download_dir = download_dir.replace("-", "_")
    download_dir = download_dir.lower()
    return download_dir


def get_biggest_gdp(n_biggest=25):
    """Order the MSA based on the GDP values, and for the n_biggest biggest MSA, select the city with biggest area"""
    gdp_df = pd.read_csv("../data/MSA_GDP.csv")[["GeoName", "2021"]]
    # remove "United States" line
    gdp_df = gdp_df[gdp_df.GeoName.apply(lambda x: x.find("United States") != 0)]

    gdp_df["GeoName"] = gdp_df.GeoName.apply(lambda x: x[: x.find(" (")])
    gdp_df["city"] = gdp_df.GeoName.apply(lambda x: x[: x.find(",")].strip().split("-"))
    gdp_df["state"] = gdp_df.GeoName.apply(
        lambda x: x[x.find(",") + 1 :].strip().split("-")
    )
    gdp_df = gdp_df.sort_values("2021", ascending=False).head(n_biggest)

    cities_shp = gpd.read_file("../data/CityBoundaries.shp")
    cities_shp["area"] = cities_shp.geometry.area
    cities_shp = cities_shp.sort_values(by="area", ascending=False)

    # create a new dataframe separating the cities of each MSA
    gdp_df_separated = []
    msa_i = 0
    for i, row in gdp_df.iterrows():
        for city in row["city"]:
            for state in row["state"]:
                filtered_shp = cities_shp[
                    ((cities_shp.NAME == city) & (cities_shp.ST == state))
                ]
                if filtered_shp.shape[0] > 0:
                    area = filtered_shp.geometry.values[0].area
                    gdp_df_separated.append([city, state, area, msa_i])
        msa_i += 1

    gdp_df_separated = pd.DataFrame(
        gdp_df_separated, columns=["city", "state", "area", "msa_i"]
    )

    # keep only the city of the biggest area for each MSA
    def keep_biggest(df):
        df = df[df.area == df.area.max()]
        return df

    gdp_df_separated = gdp_df_separated.groupby(["msa_i"]).apply(keep_biggest)
    biggest_cities = list((gdp_df_separated.city + "-" + gdp_df_separated.state).values)
    return biggest_cities


def search_scenes(
    city="New York", state="NY", dataset="naip", years=list(range(2018, 2022))
):
    """
    Function that search for all scenes of the "city-state" from the two datasets "high_res_ortho" or "naip"
    and for any year in the list "years". Return a dataframe with the scenes metadata.

    Inputs:
        city - string with the city name
        state - string with the state name
        dataset - string with the dataset name to be used, ["high_res_ortho", "naip"]
        years - list of years to select scenes

    Outputs:
        scenes_results - dataframe with columns ["dataset", "entity_id", "product_id", "spatial_coverage", "start_date", "end_date"]
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
    cities_shp["geometry"] = cities_shp["geometry"].apply(
        lambda x: MultiPolygon([x]) if type(x) == Polygon else x
    )

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
    scenes_results = pd.DataFrame(
        scenes_results,
        columns=["dataset", "entity_id", "spatial_coverage", "start_date", "end_date"],
    )
    scenes_results["geometry"] = scenes_results["spatial_coverage"].apply(
        lambda x: Polygon(x[0])
    )
    scenes_results["start_date"] = scenes_results["start_date"].apply(lambda x: x[:10])
    scenes_results["end_date"] = scenes_results["end_date"].apply(lambda x: x[:10])
    scenes_results["year"] = (
        scenes_results["start_date"].apply(lambda x: x[:4]).astype(int)
    )
    scenes_results = scenes_results[scenes_results.year.isin(years)]

    if len(scenes_results) == 0:
        return None

    # filtering scenes based on download options
    download_options = pd.DataFrame(
        m2m.downloadOptions(dataset, list(scenes_results.entity_id.values))
    )
    download_options = download_options[download_options.available]
    download_options = download_options[
        download_options.downloadSystem.isin(["dds", "dds_zip"])
    ]
    download_options = download_options[
        download_options.productName == "Full Resolution"
    ]
    download_options = download_options.drop_duplicates("entityId")
    product_id = {}
    for i, row in download_options.iterrows():
        product_id[row["entityId"]] = row["id"]

    scenes_results = scenes_results[
        scenes_results.entity_id.isin(download_options.entityId.values)
    ]
    scenes_results = scenes_results.drop_duplicates(["entity_id"])
    scenes_results["product_id"] = scenes_results.entity_id.apply(
        lambda x: product_id[x]
    )
    print(f"Total scenes for the selected years: {scenes_results.shape[0]} ")
    # convert to geodataframe and save
    scenes_results = gpd.GeoDataFrame(
        scenes_results, geometry="geometry", crs="EPSG:4326"
    )
    scenes_results = scenes_results.drop(columns=["spatial_coverage", "year"])

    return scenes_results


def download_scenes(download_dir, scenes_results):
    """
    Function that download previously searched scenes into a directory.

    Inputs:
        download_dir - string with the download directory to save scenes
        scenes_results - dataframe with scenes information (necessary columns ["entity_id", "product_id"])
    """
    scenes = scenes_results.to_dict("records")
    m2m = connect_to_m2m_api()
    m2m.retrieveScenes(scenes, download_dir)


def iterate_over_selected_cities(
    selected_cities, dataset="naip", years=list(range(2018, 2022))
):
    """
    Iterate over a list of select cities downloading their scenes from the "dataset" dataset for any of the year in "years".
    Currently not used.

    Inputs:
        selected_cities - list of strings with the format "City Name - ST"
        dataset - string with the dataset name to be used, ["high_res_ortho", "naip"]
        years - list of years to select scenes
    """
    for name in selected_cities:
        city, state = name.split("-")
        scenes_results = search_scenes(city, state, dataset, years)
        if not scenes_results is None:
            clean_city_name = get_clean_name(city)
            clean_state_name = get_clean_name(state)
            download_dir = f"./../data/output/{clean_city_name}_{clean_state_name}"
            download_scenes(download_dir, scenes_results)


def search_scenes_selected_cities_most_recent(selected_cities, dataset="naip"):
    """
    Personalized search function that will iterate over a list of selected cities and download the metadata
    of scenes for the most recent complete coverage of this city.

    Inputs:
        selected_cities - list of strings with the format "City Name - ST"
        dataset - string with dataset name, tested only with "naip"
    """
    cities_shp = gpd.read_file("../data/CityBoundaries.shp")
    cities_shp = cities_shp.to_crs("EPSG:4326")
    for name in selected_cities:
        city, state = name.split("-")
        scenes_results = search_scenes(city, state, dataset, list(range(2000, 2022)))
        if scenes_results is None:
            continue
        scenes_results["year"] = scenes_results.start_date.apply(lambda x: int(x[:4]))
        year_values = list(scenes_results.year.unique())
        year_values.sort()
        year_values.reverse()
        filtered_cities_shp = cities_shp[
            (cities_shp.NAME == city) & (cities_shp.ST == state)
        ]
        for year in year_values:
            scenes_results_year = scenes_results[scenes_results.year == year]
            if scenes_results_year.shape[0] == 0:
                continue
            if verify_scenes_covering(filtered_cities_shp, scenes_results_year):
                clean_city_name = get_clean_name(city)
                clean_state_name = get_clean_name(state)
                print(
                    f"Scenes for {name} found for year {year}: {scenes_results_year.shape[0]}"
                )
                scenes_results_year.to_file(
                    f"../data/scenes_metadata/{clean_city_name}_{clean_state_name}_last_scenes.geojson"
                )
                break


def download_scenes_selected_cities_most_recent(selected_cities):
    """
    Function that will call the main downloader function to download the most recent image of the list of selected cities.
    The function "search_scenes_selected_cities_most_recent" is necessary to be used before this function.
    It will put files into data/output/tar_files.

    Inputs:
        selected_cities - list of strings with the format "City Name - ST"
    """
    for name in selected_cities:
        city, state = name.split("-")
        clean_city_name = get_clean_name(city)
        clean_state_name = get_clean_name(state)
        scenes_results = gpd.read_file(
            f"../data/scenes_metadata/{clean_city_name}_{clean_state_name}_last_scenes.geojson"
        )
        download_dir = "./../data/output/tar_files"
        download_scenes(download_dir, scenes_results)


def verify_scenes_covering(city_shp, scenes_shp):
    """
    Verifies if the spatial extent of the scenes cover more than 85% of the area of the city

    Inputs:
        city_shp - dataframe with ["geometry"] column of city boundaries
        scenes_shp - dataframe with ["geometry"] column of scenes coverage

    Outputs:
        boolean if scenes cover city
    """
    scenes_total_cover = scenes_shp.to_crs("EPSG:32633").geometry.unary_union
    cities_total_cover = city_shp.to_crs("EPSG:32633").geometry.unary_union
    cities_total_cover = cities_total_cover.buffer(0)  # little fix
    city_intersection = cities_total_cover.intersection(scenes_total_cover)
    intersection_ratio = city_intersection.area / cities_total_cover.area
    return intersection_ratio > 0.85


def extract_tar(
    selected_files,
    output_dir="../data/output/unzipped_files",
    verify_before_extract=False,
):
    """
    Extract tar files of scene downloads into selected directory.
    It can verify if extraction already occured and not extract again.
    It also counts the number of errors that occured.

    Inputs:
        selected_files - string with the tar files names
        output_dir - string with the output directory
        verify_before_extract - boolean if should verify before extracting to not repeat processing

    """
    error = 0
    for file in tqdm(selected_files):
        entity_id = file.split("_")[0]
        try:
            with ZipFile("../data/output/tar_files/" + file, mode="r") as archive:
                files_inside_tar = archive.namelist()

                already_extracted = False
                if verify_before_extract:
                    for f in files_inside_tar:
                        if os.path.isfile(f"{output_dir}/{entity_id}_{f}"):
                            already_extracted = True

                if not already_extracted:
                    archive.extractall(output_dir)
                    for f in files_inside_tar:
                        os.rename(f"{output_dir}/{f}", f"{output_dir}/{entity_id}_{f}")
        except:
            print(f"{file} not extracted.")
            error += 1
    print(f"{error}/{len(selected_files)} files not extracted.")


def extract_tar_selected_cities_most_recent(
    selected_cities, verify_before_extract=False
):
    """
    Extract the tar files of the most recent scenes found for each city, i.e., the result of
    download_scenes_selected_cities_most_recent. Has the option to verify if it already extracted the data before.

    Inputs:
        selected_cities - list of strings in the format ["City Name-State"]
        verify_before_extract - boolean if should verify before extracting to not repeat processing
    """
    tar_files_df = pd.DataFrame(
        {
            "file_name": [
                f
                for f in os.listdir(
                    "/vida/work/GDPFinder/GDPFinder/data/output/tar_files"
                )
                if f[-4:] == ".tar"
            ]
        }
    )
    tar_files_df["entity_id"] = tar_files_df.file_name.apply(
        lambda x: int(x.split("_")[0])
    )
    for name in selected_cities:
        city, state = name.split("-")
        clean_city_name = get_clean_name(city)
        clean_state_name = get_clean_name(state)
        scenes_results = gpd.read_file(
            f"../data/scenes_metadata/{clean_city_name}_{clean_state_name}_last_scenes.geojson"
        )
        scenes_results["entity_id"] = scenes_results.entity_id.astype(int)
        tar_files_df_city = tar_files_df[
            tar_files_df.entity_id.isin(scenes_results.entity_id)
        ]
        tar_files_list_city = tar_files_df_city.file_name.tolist()
        extract_tar(tar_files_list_city, verify_before_extract=verify_before_extract)


def check_selected_cities_most_recent(selected_cities, try_download = False):
    """
    Function will verify all the files and look for errors (some missing download file or some not extracted file).
    It will return a Dataframe with the errors information.
    """
    tar_files = [
        f
        for f in os.listdir("/vida/work/GDPFinder/GDPFinder/data/output/tar_files")
        if f[-4:] == ".tar"
    ]
    tar_files_entity_id = [int(x.split("_")[0]) for x in tar_files]
    tif_files = [
        f
        for f in os.listdir("/vida/work/GDPFinder/GDPFinder/data/output/unzipped_files")
        if f[-4:] == ".tif"
    ]
    tif_files_entity_id = [int(x.split("_")[0]) for x in tif_files]
    df = []
    for name in selected_cities:
        city, state = name.split("-")
        clean_city_name = get_clean_name(city)
        clean_state_name = get_clean_name(state)
        scenes_results = gpd.read_file(
            f"../data/scenes_metadata/{clean_city_name}_{clean_state_name}_last_scenes.geojson"
        )
        scenes_results["entity_id"] = scenes_results.entity_id.astype(int)

        for i, row in scenes_results.iterrows():
            available = False
            entity_id = row["entity_id"]
            if entity_id in tar_files_entity_id and entity_id in tif_files_entity_id:
                available = True

            if not available:
                df.append([row["entity_id"], row["product_id"], city, state])

    df = pd.DataFrame(df, columns=["entity_id", "product_id", "city", "state"])

    if try_download:
        for i, row in df.iterrows():
            if row["entity_id"] in tar_files_entity_id:
                tar_filename = tar_files[tar_files_entity_id.index(row["entity_id"])]
                os.remove("../data/output/tar_files/" + tar_filename)
            if row["entity_id"] in tif_files_entity_id:
                tif_filename = tar_files[tif_files_entity_id.index(row["entity_id"])]
                os.remove("../data/output/unzipped_files/" + tif_filename)

        download_scenes("./../data/output/tar_files", df)
    return df


if __name__ == "__main__":

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    selected_cities = get_biggest_gdp(50)
    # search_scenes_selected_cities_most_recent(selected_cities)
    # download_scenes_selected_cities_most_recent(selected_cities)
    #extract_tar_selected_cities_most_recent(selected_cities, verify_before_extract=True)
    print(check_selected_cities_most_recent(selected_cities, False))
    #extract_tar_selected_cities_most_recent(selected_cities, verify_before_extract=True)
