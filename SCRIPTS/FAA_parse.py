import pandas as pd
from pathlib import Path
import json
from io import StringIO

INPUT_DIR = "./DATA/FAA"
TEST_FILE = "021204A6.txt"
OUTPUT_DIR = "./OUTPUT/FAA"


def parse_file(input_file_path):
    # read in TXT file & split the text into
    # a. metadata
    # b. actual data (starts with "Scan")

    print(f"Parsing {input_file_path}")

    with open(input_file_path, "r") as f:
        f = f.read()

    # find index of "Scan" to find where data starts
    split = f.split("\n\n\n")
    print(split[0])
    file = ("".join(split[0:2]), split[2], "".join(split[len(split) - 1]))

    # Parse the actual data
    df = pd.read_csv(StringIO(file[2]),sep="\t", header=[0, 1], on_bad_lines='skip')

    # combine columns & remove parentheses from headers

    df.columns = [" ".join([h[0].strip(), f"({h[1].strip().replace("(", "").replace(")", "")})"]) for h in df.columns]

    # remove scan (index) column
    # shifts all columns left by 1
    temp = df.columns[1:]
    df = df.drop(df.columns[len(df.columns) - 1], axis=1)
    df.columns = temp

    # df = df[["HRR (kW/m2)", "O2 (%)", "CO2 (%)", "CO (%)", "Time (secs)", "Mass (gm)"]]
    df = df[["Time (secs)", "HRR (kW/m2)", "Mass (gm)" , "O2 (%)", "CO2 (%)", "CO (%)",]]


    df = df.rename(
        columns={
            "Time (secs)": "t (s)",
            "Mass (gm)": "mass (g)",
        }
    )

    print(df)

    # create output folder
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    # output data as a CSV file
    output_name = f"{Path(input_file_path).stem}_data.csv"
    output_path = Path(OUTPUT_DIR) / output_name

    df.to_csv(output_path, index=False)

def parse_dir(input_dir):
    # get all CSV files in the input directory
    files = Path(input_dir).glob("*.txt")

    # for each file, parse it
    for file in files:
        try:
            parse_file(file)
        except:
            continue

parse_dir(INPUT_DIR)
