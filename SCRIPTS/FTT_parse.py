"""
Turns FTT cone calorimeter data into format used by NIST FCD experiments
"""

import pandas as pd
from pathlib import Path
import json

INPUT_DIR = "./DATA/FTT/"
TEST_FILE = "19010001.csv"
OUTPUT_DIR = "./OUTPUT/FTT/"


# takes in pandas dataframe (with all the FTT cone data) and extracts metadata, returning it as a Python dictionary
def extract_metadata(df):
    # metadata is found in the first two columns of the dataframe
    metadata = df[df.columns[:2]].dropna(how="all")

    # get the tranpose of the dataframe (swaps rows & columns)
    metadata = metadata.T

    # change headers to be the first row of the dataframe
    new_header = metadata.iloc[0]
    metadata = metadata[1:]
    metadata.columns = new_header

    metadata = metadata.to_dict(orient="list")
    metadata = {k: v[0] if len(v) > 0 else None for k, v in metadata.items()}

    # Remove parentheses from metadata keys, add underscores, etc.
    metadata = {process_name(k): v for k, v in metadata.items()}

    return metadata


def output_metadata(dict):
    # output metadata as a JSON file
    pass


def output_data(df):
    # output data as a CSV file
    pass


def process_name(name):
    # remove parentheses, spaces, and convert to lowercase
    name = name.replace("(", "").replace(")", "").replace(" ", "_").lower()

    # replace unicode characters with their ASCII equivalents
    name = name.replace("°", "").replace("²", "^2").replace("³", "^3")

    return name


def extract_data(df):
    # data is found in the remaining columns of the dataframe (column 3 onwards)
    """Data we care about:
    t, mass, heat release rate, O_2, CO_2, CO, K_smoke
    """
    data = df[df.columns[2:]]
    data.columns = [process_name(c) for c in data.columns]

    # we only care about certain columns, so only get that subset
    data = data[["time_s", "mass_g", "o2_%", "co2_%", "co_%"]]
    data = data.rename(
        columns={
            "time_s": "t (s)",
            "mass_g": "mass (g)",
            "o2_%": "O2 (%)",
            "co2_%": "CO2 (%)",
            "co_%": "CO (%)",
        }
    )

    return data


def parse_file(input_file_path):
    # read in CSV file as pandas dataframe
    # note the csv files are encoded in cp1252 NOT utf-8...
    df = pd.read_csv(input_file_path, encoding="cp1252")

    # metadata extracted into python dictionary (key/value pairs)
    metadata = extract_metadata(df)

    # create output folder
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    # output file name is just input file name with _metadata appended as a JSON file
    output_name = f"{Path(input_file_path).stem}_metadata.json"
    output_path = Path(OUTPUT_DIR) / output_name

    with open(output_path, "w+") as f:
        json.dump(metadata, f, indent=4)

    # extract data from dataframe
    data = extract_data(df)

    # output data as a CSV file
    output_name = f"{Path(input_file_path).stem}_data.csv"
    output_path = Path(OUTPUT_DIR) / output_name

    data.to_csv(output_path, index=False)


def parse_dir(input_dir):
    # get all CSV files in the input directory
    files = Path(input_dir).glob("*.csv")

    # for now, filter out the "red" data files
    files = filter(lambda x: "red" not in str(x), list(files))

    # for each file, parse it
    for file in files:
        parse_file(file)


# Testing code
# test_file_path = Path(INPUT_DIR, TEST_FILE).resolve()
# parse_file(test_file_path, OUTPUT_DIR)'

parse_dir(INPUT_DIR)
