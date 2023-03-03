# preprocess the metropolitan areas data

import pandas as pd
import shapefile as shp
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

import matplotlib

matplotlib.use("TKAgg")  # little fix for a problem on my env


def preprocess_pop_msa(n_cities=100):
    """It will clean the data and keep only the n_cities biggest metropolitan areas"""
    df_pop = pd.read_csv("../data/cbsa-met-est2021-pop.csv", sep=";")
    # fix columns
    df_pop["geographic_area"] = df_pop["geographic_area"].apply(
        lambda x: x[1:].replace("Metro Area", "").strip()
    )
    for col in [
        "population_census_2020",
        "population_estimate_2020",
        "population_estimate_2021",
    ]:
        df_pop[col] = df_pop[col].apply(lambda x: x.replace(".", "")).astype(float)
    # remove "sub-divisions"
    df_pop = df_pop[df_pop["geographic_area"].apply(lambda x: x[0]) != "."]
    # remove Urban Honolulu
    df_pop = df_pop[
        df_pop["geographic_area"].apply(lambda x: x.find("Urban Honolulu")) == -1
    ]
    print("Total number of metropolitan areas: ", len(df_pop))
    print("Biggest three metropolitan areas:")
    print(f"{df_pop.sort_values('population_estimate_2021', ascending=False).head(3)}")
    # save n_cities biggest metropolitans areas
    df_pop = df_pop.sort_values("population_census_2020", ascending=False).head(
        n_cities
    )
    df_pop[["geographic_area", "population_census_2020"]].to_csv(
        "../data/msa_pop.csv", index=False
    )


def preprocess_gdp_msa():
    df_gdp = pd.read_csv("../data/CAGDP1_MSA_2001_2020.csv")
    df_gdp = df_gdp[
        df_gdp.Description == "Real GDP (thousands of chained 2012 dollars)"
    ]
    df_gdp = df_gdp.drop(
        columns=[
            "Region",
            "TableName",
            "Description",
            "LineCode",
            "IndustryClassification",
            "Unit",
        ]
    )
    # create new df with a row for each city and year
    new_df_gdp = []
    for i, row in df_gdp.iterrows():
        geoid = row["GeoFIPS"]
        name = row["GeoName"]
        for year in range(2001, 2021):
            gdp = row[str(year)]
            new_df_gdp.append((geoid, name, gdp, year))
    df_gdp = pd.DataFrame(new_df_gdp, columns=["geoid", "name", "gdp", "year"])
    df_gdp["geoid"] = (
        df_gdp["geoid"].apply(lambda x: x.replace('"', "").strip()).astype(int)
    )
    df_gdp["name_clean"] = df_gdp["name"].apply(
        lambda x: x.replace("(Metropolitan Statistical Area)", "")
        .strip()
        .replace("*", "")
        .strip()
    )
    df_pop = pd.read_csv("../data/msa_pop.csv")
    df_pop["name_clean"] = df_pop["geographic_area"]
    df_gdp = pd.merge(df_pop, df_gdp, on="name_clean", how="left")
    df_gdp = df_gdp.drop(columns=["name", "name_clean", "population_census_2020"])

    n_unique_geographic_area = len(df_gdp.geographic_area.unique())
    assert n_unique_geographic_area == df_pop.shape[0]
    assert df_gdp.isna().sum().sum() == 0

    print("MSA with the three biggest GDP:")
    print(
        f"{df_gdp.groupby('geographic_area').agg({'gdp' : 'max'}).sort_values('gdp', ascending=False).head(3)}"
    )
    df_gdp.to_csv("../data/msa_gdp.csv", index=False)


def preprocess_shapefile_msa():
    df_pop = pd.read_csv("../data/msa_pop.csv")
    shp_pop = shp.Reader("../data/tl_2019_us_cbsa.shp")

    # filter shapefile with records on the pop df
    new_shp_pop = []
    for f in shp_pop.shapeRecords():
        geoid = f.record[2]
        name = f.record[4]
        name = name.replace("Metro Area", "").strip()
        if name in df_pop.geographic_area.values:
            new_shp_pop.append((name, geoid, f))

    assert len(new_shp_pop) == len(df_pop.geographic_area.values)

    # save subset shapefile
    w = shp.Writer("../data/msa_pop_biggest")
    w.field("NAME", "C")
    w.field("GEOID", "C")
    for i in range(len(new_shp_pop)):
        w.shape(new_shp_pop[i][2].shape)
        w.record(new_shp_pop[i][0], new_shp_pop[i][1])
    w.close()


def make_plots():
    df_pop = pd.read_csv("../data/msa_pop.csv")
    df_gdp = pd.read_csv("../data/msa_gdp.csv")
    shp_pop = shp.Reader("../data/msa_pop_biggest.shp")

    # plot time-series of gdp
    max_gdps = df_gdp.groupby("geographic_area").agg({"gdp": "max"}).reset_index().gdp.values
    fig, ax = plt.subplots(figsize=(14, 6))
    for name in df_gdp.geographic_area.unique():
        df = df_gdp[df_gdp.geographic_area == name]
        ax.plot(df.year, df.gdp, c = "tab:blue")
        if df.gdp.max() > sorted(max_gdps)[-4]:
            random_year = df.year.sample(1).values[0]
            ax.annotate(name[:name.find(",")], (random_year, df.gdp.max()))
    
    ax.set_xlabel("Year")
    ax.set_xticks(range(2000, 2021, 5))
    ax.set_ylabel("GDP (thousands of chained 2012 dollars)")
    ax.set_title("MSA GDP")
    plt.savefig("../figures/msa_gdp.png")

    # scatter plot gdp and pop
    fig, ax = plt.subplots(figsize=(8, 6))
    # get size of each point
    
    ax.scatter(df_pop.population_census_2020, df_gdp.groupby("geographic_area").agg({"gdp": "max"}).reset_index().gdp)
    ax.set_xlabel("Population")
    ax.set_ylabel("GDP (thousands of chained 2012 dollars)")
    ax.set_title("MSA GDP vs Population")
    plt.savefig("../figures/msa_gdp_pop.png")

    # plot values on map
    df_gdp = df_gdp.groupby("geographic_area").agg({"gdp": "max"}).reset_index()
    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(14, 6))
    cmap_1 = lambda x: plt.cm.get_cmap("Blues")(x / df_pop.population_census_2020.max())
    cmap_2 = lambda x: plt.cm.get_cmap("Greens")(x / df_gdp.gdp.max())
    for f in shp_pop.shapeRecords():
        name = f.record[0]
        pop = df_pop[df_pop.geographic_area == name].population_census_2020.values[0]
        gdp = df_gdp[df_gdp.geographic_area == name].gdp.values[0]
        p = Polygon(
            f.shape.points,
            facecolor=cmap_1(pop),
            edgecolor="#606060",
            linewidth=0.5,
        )
        axs[0].add_patch(p)

        p = Polygon(
            f.shape.points,
            facecolor=cmap_2(gdp),
            edgecolor="#606060",
            linewidth=0.5,
        )
        axs[1].add_patch(p)

    # auto scale to the bounds of the data
    for i in range(2):
        axs[i].autoscale()
        axs[i].axis("off")

    axs[0].set_title("MSA Population")
    axs[1].set_title("MSA GDP")

    plt.savefig("../figures/msa_pop.png", dpi=300)


if __name__ == "__main__":
    preprocess_pop_msa()
    preprocess_gdp_msa()
    preprocess_shapefile_msa()
    make_plots()
