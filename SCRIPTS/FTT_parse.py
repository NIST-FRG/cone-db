"""
Turns FTT cone calorimeter data into format used by NIST FCD experiments
"""

import pandas as pd
import numpy as np
from pathlib import Path


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
    metadata = {k: v[0] for k, v in metadata.items()}
    print(metadata)
    return metadata


def extract_data(df):
    # data is found in the remaining columns of the dataframe (column 3 onwards)
    data = df[df.columns[2:]]


def parse_file(path):
    # read in CSV file as pandas dataframe
    # note the csv files are encoded in cp1252 NOT utf-8...
    df = pd.read_csv(path, encoding="cp1252")
    extract_metadata(df)
    extract_data(df)


# Testing code
if __name__ == "__main__":
    DATA_DIR = "./DATA/FTT/"
    TEST_FILE = "19010001.csv"
    test_file_path = Path(DATA_DIR, TEST_FILE).resolve()

    print(test_file_path)

    parse_file(test_file_path)
