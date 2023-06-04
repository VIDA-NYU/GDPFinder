import pandas as pd
import geopandas as gpd
import requests

import os

def get_locations_info():
    locations_info = """
    eattle, wa; county code(s): 033; year: 2019
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
    !!!!! NOTICE: skipping spokane, wa due to year !!!!!
    """
    locations_info = locations_info.split("\n")
    locations_info = [x for x in locations_info if x != "" and x.find("!!!!!") == -1]
    locations_info = [x.split(";") for x in locations_info]
    locations_info = [
        {
            "city" : x[0].split(",")[0],
            "state" : x[0].split(",")[1].strip(" "),
            "county_codes" : x[1].split(":")[1].strip(" "),
            "year" : x[2].split(":")[1].strip(" ")
        }
        for x in locations_info
    ]
    locations_info = pd.DataFrame(locations_info)
    return locations_info

def get_states_codes():
    states_codes = """
    Alabama	01	AL
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
    Wyoming	56	WY
    """
    states_codes = states_codes.split("\n")
    states_codes = [x for x in states_codes if x != ""]
    states_codes = [x.split("\t") for x in states_codes]
    states_codes = [
        {
            "state" : x[0].lower().replace(" ", "_"),
            "state_code" : x[1],
            "state_abbr" : x[2].lower()
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
        state_code = states_codes[states_codes["state_abbr"] == state_abbr]["state_code"].values[0]
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
    census_df["ed_attain"] = (100*((census_df["B15003_022E"] + census_df["B15003_023E"] + census_df["B15003_024E"] + census_df["B15003_025E"]) / census_df["B15003_001E"])).round(2)

    # Remove columns
    census_df.drop(["B15003_001E", "B15003_022E", "B15003_023E", "B15003_024E","B15003_025E"], axis=1, inplace=True)


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
    scenes_df = [gpd.read_file("../data/scenes_metadata/"+f) for f in os.listdir("../data/scenes_metadata/")]
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
    blocks_df = blocks_df.rename(columns = {"STATEFP" : "state", "COUNTYFP" : "county", "TRACTCE" : "tract", "BLKGRPCE" : "block group"})
    blocks_df = blocks_df[['state', 'county', 'tract', 'block group', 'geometry']]
    return blocks_df

if __name__ == "__main__":
    census_df = request_census_data()
    blocks_df = get_blocks_df()
    blocks_df = blocks_df.merge(census_df, on=["state", "county", "tract", "block group"], how="left")
    blocks_df.to_file("../data/census_blocks.geojson")


