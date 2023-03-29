import pandas as pd
import geopandas as gpd
import shapefile as shp
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import matplotlib

#matplotlib.use("TKAgg")  # little fix for a problem on my env


def get_biggest_city_from_biggest_gdp_msa():
    """Order the MSA based on the GDP values, and for the n_biggest biggest MSA, select the city with biggest area"""
    gdp_df = pd.read_csv("../data/MSA_GDP.csv")[["GeoName", "2021"]]
    # remove "United States" line
    gdp_df = gdp_df[gdp_df.GeoName.apply(lambda x: x.find("United States") != 0)]

    gdp_df["GeoName"] = gdp_df.GeoName.apply(lambda x: x[: x.find(" (")])
    gdp_df["city"] = gdp_df.GeoName.apply(lambda x: x[: x.find(",")].strip().split("-"))
    gdp_df["state"] = gdp_df.GeoName.apply(
        lambda x: x[x.find(",") + 1 :].strip().split("-")
    )
    gdp_df = gdp_df.sort_values("2021", ascending=False)

    cities_shp = gpd.read_file("../data/CityBoundaries.shp")
    cities_shp["area"] = cities_shp.geometry.area

    # create a new dataframe separating the cities of each MSA
    gdp_df_separated = []
    for i, row in gdp_df.iterrows():
        for city in row["city"]:
            for state in row["state"]:
                filtered_shp = cities_shp[
                    ((cities_shp.NAME == city) & (cities_shp.ST == state))
                ]
                if filtered_shp.shape[0] > 0:
                    area = filtered_shp.geometry.values[0].area
                    gdp_df_separated.append([city, state, area, row["GeoName"], row["2021"]])

    gdp_df_separated = pd.DataFrame(
        gdp_df_separated, columns=["city", "state", "area", "msa_name", "gdp_2021"]
    )

    # keep only the city of the biggest area for each MSA
    def keep_biggest(df):
        df = df[df.area == df.area.max()]
        return df

    gdp_df_separated = gdp_df_separated.groupby(["msa_name"]).apply(keep_biggest).reset_index(drop = True)
    gdp_df_separated = gdp_df_separated.sort_values("gdp_2021", ascending = False)
    return gdp_df_separated


if __name__ == "__main__":
    df = get_biggest_city_from_biggest_gdp_msa()
    df.to_csv("../data/cities_biggest_gdp.csv", index = False)

