import pandas as pd
from pathlib import Path
import json
import sys
import re

from utils import calculate_HRR, calculate_MFR, calculate_k, colorize

args = sys.argv[1:]
if len(args) > 2:
    print(
        """Too many arguments
          Usage: python parse-legacy.py <input_dir> <output_dir>
          Leave empty to use defaults."""
    )
    sys.exit()

# these assume the script is being run from the root of the repo
INPUT_DIR = Path(r"./data/raw/legacy")
OUTPUT_DIR = Path(r"./data/auto-processed/legacy")

if len(args) == 2:
    INPUT_DIR = Path(args[0])
    OUTPUT_DIR = Path(args[1])


def parse_dir(input_dir):
    # read all MD files in the input directory
    paths = Path(input_dir).glob("**/*.md")

    # open the folder for each format and read the files using the appropriate function
    for path in paths:
        print(f"Parsing file at {path}, ", end="")
        if path.parts[-2] == "A":
            print("format A")
            read_format_A(path)
        else:
            print(f"Unknown format {path.stem}, skipping")
            continue


def read_format_A(path):
    def read_data(lines):
        for i in range(len(lines)):
            if re.search(r"Page \d+ of \d+", lines[i]):
                print("Start of data found, line: ", lines[i])
                lines = lines[i + 1 :]
                break

        # second line is always the header
        header = lines[1].split("|")
        header = [x.strip() for x in header if x.strip() != ""]
        units = lines[2].split("|")
        units = [x.strip() for x in units if x.strip() != ""]
        # set header from the line as the header of the df
        df = pd.DataFrame(columns=header)

        for j in range(len(lines)):
            line = lines[j]

            if not (line.strip().startswith("|") and line.strip().endswith("|")):
                continue

            # if the line contains any of the values in the header or the units, it does not contain data
            if any([x in line for x in header + units]):
                continue

            # split line up into different values
            values = line.strip().split("|")
            values = [x.strip() for x in values if x.strip() != ""]

            # if these lines contain only dashes (in any number), skip them:
            if all([re.match(r"-+", x) for x in values]):
                continue

            # if the line contains text of any kind, skip it
            if any([re.search(r"[a-zA-Z]", x) for x in values]):
                continue

            # if, at any point, the number of values is not equal to the number of headers, skip the line
            if len(values) != len(header):
                continue

            # replace any strings containing only stars (e.g. "****") with None
            values = [None if re.match(r"\*+", x) else x for x in values]

            # add the values to the dataframe, row by row
            new_row = pd.DataFrame([values], columns=header)
            df = pd.concat([df, new_row], ignore_index=True)

        # rename the columns of the df
        df = df.rename(
            columns={
                "Time": "Time (s)",
                "Q-Dot": "HRR (kW/m2)",
            }
        )

        print(df.head())
        return df

    def read_metadata(lines):
        metadata = {}

        def extract_string(key, input_str, end=""):
            if end == "":
                match = re.search(key + r"\s*:\s*(.+)\n", input_str)
            else:
                match = re.search(
                    key + r"\s*:\s*(.+)\n+\s*" + end, input_str, flags=re.DOTALL
                )
            if match:
                return match.group(1).strip()
            else:
                return None

        input_str = "\n".join(lines)

        metadata["comments"] = extract_string(
            "PRE-TEST COMMENTS", input_str, end="Mass ratio average"
        )
        metadata["orientation"] = (
            "Horizontal"
            if "Horizontal" in lines[3]
            else "Vertical" if "Vertical" in lines[3] else None
        )
        metadata["surface_area_cm2"] = extract_string(
            "Area of Sample", input_str, end="m2"
        )
        metadata["c_factor"] = extract_string(
            "Conversion Factor", input_str, end="kJoule/kg"
        )

        return metadata

    def write_metadata(name, metadata):
        with open(OUTPUT_DIR / f"{name}.json", "w+") as f:
            json.dump(metadata, f, indent=4)

    def write_data(name, df):
        df.to_csv(OUTPUT_DIR / f"{name}.csv", index=False)

    with open(path, "r") as f:
        lines = list(f.readlines())

        df = read_data(lines)
        metadata = read_metadata(lines)

        name = path.stem

        Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

        write_metadata(name, metadata)
        write_data(name, df)


parse_dir(INPUT_DIR)
