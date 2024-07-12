import pandas as pd
from pathlib import Path
import json
from dateutil import parser

INPUT_DIR = "./DATA/MIDAS"
OUTPUT_DIR = "./OUTPUT/MIDAS"
DEUBG = True


def parse_dir(input_dir):
    paths = Path(input_dir).glob("**/*scaled.csv")

    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    for path in paths:
        try:
            parse_file(path)
        except Exception as e:
            print(f"Error parsing {path}: {e}")


def parse_file(file_path):
    print(f"Parsing {file_path}")

    # read in scaled CSV file as pd dataframe
    df = pd.read_csv(file_path, encoding="cp1252")

    data = parse_data(df)

    metadata = parse_metadata(file_path)

    metadata_output_path = Path(OUTPUT_DIR) / f"{Path(file_path).stem.replace("-scaled.csv", "_metadata.json")}_metadata.json"

    # write metadata to json file
    with open(metadata_output_path, "w") as f:
        json.dump(metadata, f, indent=4)


def parse_metadata(file_path):
    # get the -Output .xls file for the metadata
    file_path = str(file_path).replace("-scaled.csv", "-Output.xls")

    events = pd.read_excel(file_path, sheet_name="User Events")
    params = pd.read_excel(file_path, "Parameters")
    info = pd.read_excel(file_path, "Info", header=None)

    # convert params to a dictionary
    params = params.to_dict(orient="list")
    params = {k: v[0] if len(v) > 0 else None for k, v in params.items()}

    def get_number(key, dict):
        try:
            return float(dict[key])
        except:
            return None

    def get_bool(key, params):
        string = params[key].lower()
        if string == "no" or string == "false":
            return False
        elif string == "yes" or string == "true":
            return True
        return None

    metadata = {}

    # Get test parameters
    metadata["c_factor"] = get_number("Cf", params)
    metadata["e_mj/kg"] = get_number("Ef", params) / 1000
    metadata["heat_flux_kw/m2"] = get_number("CONEHEATFLUX", params)
    metadata["grid"] = get_bool("Grid", params)
    metadata["separation_mm"] = get_number("Separation", params)
    metadata["initial_mass_g"] = get_number("ISMass", params)
    metadata["orientation"] = params["ORIENTATION"]

    # Get test info

    # get the tranpose of the dataframe (swaps rows & columns)
    info = info.T
    
    new_header = info.iloc[0]
    info = info[1:]
    info.columns = new_header
    # transform into a dictionary
    info = info.to_dict(orient="list")

    # if there are multiple values, take the first one
    info = {k: v[0] if len(v) > 0 else None for k, v in info.items()}
    info = {k.replace(":", ""): v for k, v in info.items()}

    # parse dates
    date = parser.parse(f"{info["Date"]} {info["Time"]}", dayfirst=True)

    metadata["date"] = date.isoformat()
    metadata["operator"] = info["Qualified Operator"]
    metadata["comments"] = info["Test Series information"]
    metadata["specimen_number"] = info["Sample ID"]

    # events
    def parse_event(row):
        return {
            "time": row["Time (s)"],
            "event": row["Event Description"],
        }
    
    parsed_events = [parse_event(row) for _, row in events.iterrows()]

    metadata["events"] = parsed_events

    # use event info to set ignition time, burnout, start, etc.
    for event in parsed_events:
        if "Ignition" in event["event"]:
            metadata["time_to_ignition_s"] = event["time"]
        elif "Flame Out" in event["event"]:
            metadata["time_to_flameout_s"] = event["time"]
        elif "Start" in event["event"]:
            metadata["test_start_time_s"] = event["time"]

    return metadata


def parse_data(df):
    print(df)
    data = df[["Time (s)", "O2 (Vol fr)", "CO2 (Vol fr)", "CO (Vol fr)", ]]