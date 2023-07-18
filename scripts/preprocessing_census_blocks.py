import numpy as np
import pandas as pd
import geopandas as gpd
import requests
from sklearn.neighbors import KDTree
from tqdm import tqdm
import shapely
import os
from ast import literal_eval
from sklearn.model_selection import train_test_split


def get_locations_info():
    locations_info = """seattle, wa; county code(s): 033; year: 2019
new_york, ny; county code(s): 005,047,061,081,085; year: 2021
los_angeles, ca; county code(s): 037; year: 2020
chicago, il; county code(s): 043,031; year: 2021
oakland, ca; county code(s): 001; year: 2020
dallas, tx; county code(s): 121,397,113,257,085; year: 2020
washington, dc; county code(s): 001; year: 2021
houston, tx; county code(s): 157,339,201,473; year: 2020
boston, ma; county code(s): 025; year: 2021
philadelphia, pa; county code(s): 101; year: 2019
atlanta, ga; county code(s): 089,121; year: 2019
san_jose, ca; county code(s): 085; year: 2020
!!!!! NOTICE: skipping miami, fl due to year !!!!!
phoenix, az; county code(s): 013; year: 2021
minneapolis, mn; county code(s): 053; year: 2021
detroit, mi; county code(s): 163; year: 2020
san_diego, ca; county code(s): 073; year: 2020
aurora, co; county code(s): 001,005,035; year: 2019
baltimore, md; county code(s): 510; year: 2021
riverside, ca; county code(s): 065; year: 2020
charlotte, nc; county code(s): 119; year: 2020
austin, tx; county code(s): 491,209,453,021; year: 2020
portland, or; county code(s): 051,005,067; year: 2020
tampa, fl; county code(s): 057; year: 2021
st._louis, mo; county code(s): 510; year: 2020
cincinnati, oh; county code(s): 061; year: 2021
pittsburgh, pa; county code(s): 003; year: 2019
orlando, fl; county code(s): 095; year: 2021
nashville, tn; county code(s): 037; year: 2021
indianapolis, in; county code(s): 097; year: 2020
sacramento, ca; county code(s): 067; year: 2020
kansas_city, mo; county code(s): 037,047,165,095; year: 2020
columbus, oh; county code(s): 049,045,041; year: 2021
san_antonio, tx; county code(s): 029,325,091; year: 2020
cleveland, oh; county code(s): 035; year: 2021
las_vegas, nv; county code(s): 003; year: 2019
salt_lake_city, ut; county code(s): 035; year: 2021
milwaukee, wi; county code(s): 131,079,133; year: 2020
raleigh, nc; county code(s): 063,183; year: 2020
durham, nc; county code(s): 135,063,183; year: 2020
hartford, ct; county code(s): 110; year: 2021
!!!!! NOTICE: error for hartford, ct: Expecting value: line 1 column 1 (char 0) !!!!!
virginia_beach, va; county code(s): 810; year: 2021
jacksonville, fl; county code(s): 031,089; year: 2021
richmond, va; county code(s): 760; year: 2021
warwick, ri; county code(s): 003; year: 2021
oklahoma_city, ok; county code(s): 125,027,109,017; year: 2021
stamford, ct; county code(s): 190; year: 2021
!!!!! NOTICE: error for stamford, ct: Expecting value: line 1 column 1 (char 0) !!!!!
memphis, tn; county code(s): 157; year: 2021
new_orleans, la; county code(s): 071; year: 2021
buffalo, ny; county code(s): 029; year: 2021
omaha, ne; county code(s): 055; year: 2020
albany, ny; county code(s): 001; year: 2021
birmingham, al; county code(s): 073,117; year: 2021
rochester, ny; county code(s): 055; year: 2021
grand_rapids, mi; county code(s): 081; year: 2020
tulsa, ok; county code(s): 131,143,145,113; year: 2021
des_moines, ia; county code(s): 153,181; year: 2021
baton_rouge, la; county code(s): 033; year: 2021
thousand_oaks, ca; county code(s): 111; year: 2020
madison, wi; county code(s): 025; year: 2020
new_haven, ct; county code(s): 170; year: 2021
!!!!! NOTICE: error for new_haven, ct: Expecting value: line 1 column 1 (char 0) !!!!!
bakersfield, ca; county code(s): 029; year: 2020
worcester, ma; county code(s): 027; year: 2021
knoxville, tn; county code(s): 093; year: 2021
bethlehem, pa; county code(s): 077,095; year: 2019
fresno, ca; county code(s): 019; year: 2020
charleston, sc; county code(s): 015,019; year: 2021
tucson, az; county code(s): 019; year: 2021
dayton, oh; county code(s): 057,113; year: 2021
albuquerque, nm; county code(s): 001; year: 2020
columbia, sc; county code(s): 079,063; year: 2021
midland, tx; county code(s): 329,317; year: 2020
syracuse, ny; county code(s): 067; year: 2021
greensboro, nc; county code(s): 081; year: 2020
colorado_springs, co; county code(s): 041; year: 2019
boise_city, id; county code(s): 001; year: 2021
trenton, nj; county code(s): 021; year: 2019
little_rock, ar; county code(s): 119; year: 2021
toledo, oh; county code(s): 095; year: 2021
wichita, ks; county code(s): 173; year: 2021
akron, oh; county code(s): 153; year: 2021
portland, me; county code(s): 005; year: 2021
cape_coral, fl; county code(s): 071; year: 2021
provo, ut; county code(s): 049; year: 2021
el_paso, tx; county code(s): 141; year: 2020
springfield, ma; county code(s): 013; year: 2021
stockton, ca; county code(s): 077; year: 2020
ogden, ut; county code(s): 057; year: 2021
boulder, co; county code(s): 013; year: 2019
huntsville, al; county code(s): 103,089,083; year: 2021
santa_maria, ca; county code(s): 083; year: 2020
reno, nv; county code(s): 031; year: 2019
santa_rosa, ca; county code(s): 097; year: 2020
chattanooga, tn; county code(s): 065; year: 2021
fayetteville, ar; county code(s): 143; year: 2021
lexington, ky; county code(s): 067; year: 2020
manchester, nh; county code(s): 011; year: 2021
lakeland, fl; county code(s): 105; year: 2021
vallejo, ca; county code(s): 095; year: 2020
!!!!! NOTICE: skipping spokane, wa due to year !!!!!"""
    locations_info = locations_info.split("\n")
    locations_info = [x for x in locations_info if x != "" and x.find("!!!!!") == -1]
    locations_info = [x.split(";") for x in locations_info]
    locations_info = [
        {
            "city": x[0].split(",")[0],
            "state": x[0].split(",")[1].strip(" "),
            "county_codes": x[1].split(":")[1].strip(" "),
            "year": x[2].split(":")[1].strip(" "),
        }
        for x in locations_info
    ]
    locations_info = pd.DataFrame(locations_info)
    return locations_info


def get_states_codes():
    states_codes = """Alabama	01	AL
Alaska	02	AK
Arizona	04	AZ
Arkansas	05	AR
California	06	CA
Colorado	08	CO
Connecticut	09	CT
Delaware	10	DE
District of Columbia	11	DC
Florida	12	FL
Georgia	13	GA
Hawaii	15	HI
Idaho	16	ID
Illinois	17	IL
Indiana	18	IN
Iowa	19	IA
Kansas	20	KS
Kentucky	21	KY
Louisiana	22	LA
Maine	23	ME
Maryland	24	MD
Massachusetts	25	MA
Michigan	26	MI
Minnesota	27	MN
Mississippi	28	MS
Missouri	29	MO
Montana	30	MT
Nebraska	31	NE
Nevada	32	NV
New Hampshire	33	NH
New Jersey	34	NJ
New Mexico	35	NM
New York	36	NY
North Carolina	37	NC
North Dakota	38	ND
Ohio	39	OH
Oklahoma	40	OK
Oregon	41	OR
Pennsylvania	42	PA
Rhode Island	44	RI
South Carolina	45	SC
South Dakota	46	SD
Tennessee	47	TN
Texas	48	TX
Utah	49	UT
Vermont	50	VT
Virginia	51	VA
Washington	53	WA
West Virginia	54	WV
Wisconsin	55	WI
Wyoming	56	WY"""
    states_codes = states_codes.split("\n")
    states_codes = [x for x in states_codes if x != ""]
    states_codes = [x.split("\t") for x in states_codes]
    states_codes = [
        {
            "state": x[0].lower().replace(" ", "_"),
            "state_code": x[1],
            "state_abbr": x[2].lower(),
        }
        for x in states_codes
    ]
    states_codes = pd.DataFrame(states_codes)
    states_codes
    return states_codes


def request_census_data():
    locations_info = get_locations_info()
    states_codes = get_states_codes()
    ## Retrieve census data for specified location(s) and geography type (e.g., block group)

    # Replace YOUR_API_KEY_HERE with your own API key
    API_KEY = "bd041cbc149095f7871a9d002c6ca41d7b8010ef"

    # Define the API endpoint
    url = "https://api.census.gov/data/{year}/acs/acs5"

    # Define the variables and geography of interest
    variables = "B01003_001E,B19013_001E,B15003_001E,B15003_022E,B15003_023E,B15003_024E,B15003_025E"
    geography = "block group:*"
    requests_df = []
    for i, location in locations_info.iterrows():
        year = location["year"]
        state_abbr = location["state"]
        state_code = states_codes[states_codes["state_abbr"] == state_abbr][
            "state_code"
        ].values[0]
        county_sequence = location["county_codes"]
        location = f"state:{state_code}+county:{county_sequence}+tract:*"

        # Build the API query
        query = f"{url.format(year=year)}?get={variables}&for={geography}&in={location}&key={API_KEY}"

        try:
            # Make the API call
            response = requests.get(query)

            # Parse the response as JSON
            acs_data = response.json()

            # Convert the list to a DataFrame
            temp_df = pd.DataFrame(acs_data[1:], columns=acs_data[0])
            # Add the year column to the DataFrame
            temp_df["year"] = year
            requests_df.append(temp_df)

        except requests.exceptions.JSONDecodeError as err:
            print(f"!!!!! NOTICE: error for {query}: {err} !!!!!")
    # Concatenate all DataFrames in the list
    census_df = pd.concat(requests_df)

    # Calculate % of population over 25 years old with a bachelors degree and add to df; rename and remove columns
    census_df["B15003_001E"] = pd.to_numeric(census_df["B15003_001E"])
    census_df["B15003_022E"] = pd.to_numeric(census_df["B15003_022E"])
    census_df["B15003_023E"] = pd.to_numeric(census_df["B15003_023E"])
    census_df["B15003_024E"] = pd.to_numeric(census_df["B15003_024E"])
    census_df["B15003_025E"] = pd.to_numeric(census_df["B15003_025E"])
    census_df["ed_attain"] = (
        100
        * (
            (
                census_df["B15003_022E"]
                + census_df["B15003_023E"]
                + census_df["B15003_024E"]
                + census_df["B15003_025E"]
            )
            / census_df["B15003_001E"]
        )
    ).round(2)

    # Remove columns
    census_df.drop(
        ["B15003_001E", "B15003_022E", "B15003_023E", "B15003_024E", "B15003_025E"],
        axis=1,
        inplace=True,
    )

    # Rename columns
    census_df.rename(columns={"B01003_001E": "pop"}, inplace=True)
    census_df.rename(columns={"B19013_001E": "mhi"}, inplace=True)
    census_df["pop"] = pd.to_numeric(census_df["pop"])
    census_df["mhi"] = pd.to_numeric(census_df["mhi"])
    census_df["ed_attain"] = pd.to_numeric(census_df["ed_attain"])
    census_df = census_df.drop_duplicates()
    return census_df


def get_blocks_df():
    block_folders = os.listdir("../data/blocks")
    scenes_df = [
        gpd.read_file("../data/scenes_metadata/" + f)
        for f in os.listdir("../data/scenes_metadata/")
    ]
    scenes_df = gpd.GeoDataFrame(pd.concat(scenes_df, ignore_index=True))
    scenes_df_poly = scenes_df.unary_union
    blocks_df = []
    for bf in block_folders:
        filename = bf + ".shp"
        block_df = gpd.read_file(f"../data/blocks/{bf}/{filename}")
        block_df = block_df[block_df.geometry.intersects(scenes_df_poly)]
        if len(block_df) > 0:
            blocks_df.append(block_df)
    blocks_df = gpd.GeoDataFrame(pd.concat(blocks_df, ignore_index=True))
    blocks_df = blocks_df.rename(
        columns={
            "STATEFP": "state",
            "COUNTYFP": "county",
            "TRACTCE": "tract",
            "BLKGRPCE": "block group",
        }
    )
    blocks_df = blocks_df[["state", "county", "tract", "block group", "geometry"]]
    return blocks_df


def get_patches_inside_blocks(patches, blocks):
    patches = patches.to_crs("epsg:3395")
    blocks = blocks.to_crs("epsg:3395")

    # build tree with patches centers
    patches_centers = np.stack(
        patches.geometry.apply(lambda x: np.array(x.centroid.coords)).values
    ).squeeze()
    tree = KDTree(patches_centers)

    patch_area = patches.area.mean()
    relation = []
    # for each patch
    for i, row in tqdm(blocks.iterrows(), total=blocks.shape[0]):
        block_area = row["geometry"].area
        # estimate a good number of neighbors to search for
        k = int(max(5, min(block_area // patch_area, patches.shape[0] / 5)))
        # verify if it intersects the k closest patches
        centroid = np.array(row["geometry"].centroid.coords).reshape(1, 2)
        idx_closest = tree.query(centroid, k=k)[1][0]
        intersection_ratio = (
            patches.iloc[idx_closest].geometry.intersection(row.geometry).area
            / patch_area
        ).values

        block_patches = {}
        # for each of the closest patches
        for idx, ratio in zip(idx_closest, intersection_ratio):
            if ratio == 0:
                continue

            # saves the filename and the ratio of intersection
            filename = patches.iloc[idx].patche_filename
            if filename in block_patches.keys():
                block_patches[filename].append(
                    [patches.iloc[idx].idx, np.round(ratio, 3)]
                )
            else:
                block_patches[filename] = []
                block_patches[filename].append(
                    [patches.iloc[idx].idx, np.round(ratio, 3)]
                )

        block_patches = str(block_patches)
        relation.append(block_patches)

    return relation


def split_rectangle(rect):
    x0, y0, x1, y1 = rect.bounds
    x_mid = (x0 + x1) / 2
    y_mid = (y0 + y1) / 2
    return [
        shapely.geometry.box(x0, y0, x_mid, y_mid),
        shapely.geometry.box(x_mid, y0, x1, y_mid),
        shapely.geometry.box(x0, y_mid, x_mid, y1),
        shapely.geometry.box(x_mid, y_mid, x1, y1),
    ]


def split_patches_df(patches_df):
    new_df = []
    for i, row in patches_df.iterrows():
        new_geom = split_rectangle(row["geometry"])
        for j in range(4):
            new_row = row.copy()
            new_row["geometry"] = new_geom[j]
            new_row["idx"] = j
            new_df.append(new_row)
    new_df = gpd.GeoDataFrame(new_df)
    new_df = new_df.set_crs(patches_df.crs)
    return new_df


def compute_blocks_and_patches_relation():
    blocks_df = gpd.read_file("../data/census_blocks.geojson")

    # get name of cities shapefiles
    cities_shp = os.listdir("../data/scenes_metadata/")
    cities_shp = [s for s in cities_shp if s.endswith(".geojson")]
    blocks_df["patches_relation"] = ""

    # going to compare patches and blocks separated by city for better computing time
    for city_shp in cities_shp:
        city_df = gpd.read_file("../data/scenes_metadata/" + city_shp)
        city_state = city_shp.replace("_last_scenes.geojson", "")

        # keep only blocks inside city
        is_in_city = blocks_df.geometry.intersects(city_df.unary_union)
        blocks_of_city = blocks_df[is_in_city]

        # get patches of the city
        patches_of_city = os.listdir("../data/output/patches/" + city_state)
        patches_of_city = [s for s in patches_of_city if s.endswith(".geojson")]
        patches_of_city = gpd.GeoDataFrame(
            pd.concat(
                [
                    gpd.read_file("../data/output/patches/" + city_state + "/" + s)
                    for s in patches_of_city
                ]
            )
        )
        patches_of_city["patche_filename"] = (
            city_state + "/" + patches_of_city.patche_filename
        )
        patches_of_city = split_patches_df(patches_of_city)
        # run function that identify the relation between them
        relation = get_patches_inside_blocks(patches_of_city, blocks_of_city)

        blocks_df.loc[is_in_city, "patches_relation"] = relation
        # remove geometry and save to csv
        blocks_df.drop("geometry", axis=1).to_csv("../data/blocks_patches_relation.csv")
        # blocks_df.to_file("../data/census_blocks_patches_v3.geojson")

def create_train_test_df(intersection_threshold = 0.25, patches_count_max = 100):
    blocks_df = pd.read_csv("../data/blocks_patches_relation.csv")
    blocks_df["mhi"] = blocks_df["mhi"].apply(lambda x: np.nan if x < 0 else x)
    blocks_df["patches_relation"] = blocks_df["patches_relation"].apply(lambda x : np.nan if x == "{}" else x)
    blocks_df = blocks_df.dropna() 
    blocks_df["patches_relation"] = blocks_df["patches_relation"].apply(literal_eval)
    blocks_df["n_patches"] = blocks_df["patches_relation"].apply(len)
      
    def get_n_scenes(x):
        return len(set([f.split("/")[1].split("_")[0] for f in x.keys()]))
    def get_most_commom_scene(x):
        scenes_id = [f.split("/")[1].split("_")[0] for f in x.keys()]
        u, c = np.unique(scenes_id, return_counts=True)
        return u[np.argmax(c)]

    blocks_df["n_scenes"] = blocks_df.patches_relation.apply(get_n_scenes)
    blocks_df["most_commom_scene"] = blocks_df.patches_relation.apply(get_most_commom_scene)
    for i, row in blocks_df.iterrows():
        blocks_df.at[i, "patches_relation"] = {k: v for k, v in row.patches_relation.items() if k.split("/")[1].split("_")[0] == row.most_commom_scene}
    blocks_df.n_patches = blocks_df.patches_relation.apply(len)
    blocks_df = blocks_df[blocks_df.n_patches > 0]
    print(f"Total of {blocks_df.shape[0]} blocks")

    # filter patches based on intersection threshold and max number of patches
    all_filenames = []
    for i, row in tqdm(blocks_df.iterrows(), total=blocks_df.shape[0]):
        s = row.patches_relation
        filenames = []
        data = []
        # transform test into array, filtering by intersection threshold
        for key, value in s.items():
            value = np.array(value)
            value = value[value[:, 1] > intersection_threshold, :]
            for i in range(len(value)):
                filenames.append(key)
                data.append(value[i, :])
        data = np.array(data)
        if len(filenames) > patches_count_max:
            selected = np.random.choice(
                len(filenames),
                patches_count_max,
                replace=False,
                p=data[:, 1] / data[:, 1].sum(),
            )
            data = data[selected, :]
            filenames = [filenames[i] for i in selected]
            filenames = [f"../data/output/patches/{filenames[i]} {int(data[i, 0])}" for i in range(len(filenames))]
        all_filenames.append(filenames)
    blocks_df["filenames"] = all_filenames

    # split train and test
    train_idx, test_idx = train_test_split(blocks_df.index, test_size=0.15, random_state=42)
    train_idx, val_idx = train_test_split(train_idx, test_size=0.15, random_state=42)
    blocks_df_train = blocks_df.loc[train_idx]
    blocks_df_train.to_csv("../data/blocks_patches_relation_train.csv", index=False)
    blocks_df_val = blocks_df.loc[val_idx]
    blocks_df_val.to_csv("../data/blocks_patches_relation_val.csv", index=False)
    blocks_df_test = blocks_df.loc[test_idx]
    blocks_df_test.to_csv("../data/blocks_patches_relation_test.csv", index=False)
    print(f"Train: {len(train_idx)} Files: {blocks_df_train.n_patches.sum()}")
    print(f"Val: {len(val_idx)} Files: {blocks_df_val.n_patches.sum()}")
    print(f"Test: {len(test_idx)} Files: {blocks_df_test.n_patches.sum()}")

if __name__ == "__main__":
    # census_df = request_census_data()
    # blocks_df = get_blocks_df()
    # blocks_df = blocks_df.merge(census_df, on=["state", "county", "tract", "block group"], how="left")
    # blocks_df = gpd.read_file("../data/census_blocks.geojson")
    # blocks_df["area"] = blocks_df.geometry.to_crs({"proj": "cea"}).area / 10**6
    # blocks_df["density"] = blocks_df["pop"] / blocks_df["area"]
    # blocks_df.to_file("../data/census_blocks.geojson")
    # compute_blocks_and_patches_relation()
    create_train_test_df()
