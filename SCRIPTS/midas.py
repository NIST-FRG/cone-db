import pandas as pd
from pathlib import Path
import json
import matplotlib.pyplot as plt
from dateutil import parser

from processing import calculate_HRR, calculate_MFR


INPUT_DIR = "./DATA/MIDAS"
OUTPUT_DIR = "./OUTPUT/MIDAS"
DEUBG = True


def parse_dir(input_dir):
    paths = Path(input_dir).glob("**/*scaled.csv")

    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    for path in paths:
        # try:
        parse_file(path)
        # except Exception as e:
        #     print(f"Error parsing {path}: {e}")


def parse_file(file_path):
    print(f"Parsing {file_path}")

    # read in scaled CSV file as pd dataframe
    df = pd.read_csv(file_path, encoding="cp1252")

    metadata = parse_metadata(file_path)

    metadata_output_path = Path(OUTPUT_DIR) / f"{Path(file_path).stem.replace("-scaled.csv", "_metadata.json")}_metadata.json"

    # write metadata to json file
    with open(metadata_output_path, "w") as f:
        json.dump(metadata, f, indent=4)

    data = parse_data(df, metadata)

    data_output_path = Path(OUTPUT_DIR) / f"{Path(file_path).stem}_data.csv"
    data.to_csv(data_output_path, index=False)




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


def parse_data(df, metadata):

    # Convert Te into K
    df["25: Te (°C)"] += 273.15

    data = df.rename(
        columns={
            "Test Time (s)": "Time (s)",
            "0: O2 (Vol fr)": "O2 (Vol fr)",
            "1: CO2 (Vol fr)": "CO2 (Vol fr)",
            "2: CO (Vol fr)": "CO (Vol fr)",
            "25: Te (°C)": "Te (K)",
            "5: Pe (Pa)": "Pe (Pa)",
        }
    )

    data = process_data(data, metadata)

    
    data = data[["Time (s)", "O2 (Vol fr)", "CO2 (Vol fr)", "CO (Vol fr)", "HRR (kW/m2)", "MFR (kg/s)"]]

    return data

def process_data(data, metadata):

    start_time = metadata.get("test_start_time_s", 0)
    o2_delay = metadata.get("o2_delay_time_s", 0)
    co2_delay = metadata.get("co2_delay_time_s", 0)
    co_delay = metadata.get("co_delay_time_s", 0)
    area = metadata.get("surface_area_cm^2", 100)
    c_factor = metadata.get("c_factor")
    e = metadata.get("e_mj/kg", 13.1)

    # convert area to m^2
    area = area / (100**2)

        # calculate baseline values by using the data up to test start time
    X_O2_initial = data["O2 (Vol fr)"][:start_time].mean() / 100
    X_CO2_initial = data["CO2 (Vol fr)"][:start_time].mean() / 100
    X_CO_initial = data["CO (Vol fr)"][:start_time].mean() / 100

    # shift entire dataframe up to start time
    data = data.shift(-start_time)
    data.drop(data.tail(start_time).index, inplace=True)

    data["Time (s)"] -= start_time

    # shift certain columns up to account for O2, CO, CO2 analyzer time delay, and remove the rows at the end
    data["O2 (Vol fr)"] = data["O2 (Vol fr)"].shift(-o2_delay)
    data["CO2 (Vol fr)"] = data["CO2 (Vol fr)"].shift(-co2_delay)
    data["CO (Vol fr)"] = data["CO (Vol fr)"].shift(-co_delay)

    data.drop(data.tail(max(o2_delay, co_delay, co2_delay)).index, inplace=True)

    # Calculate HRR by row

    def get_HRR(row):
        X_O2 = row["O2 (Vol fr)"] / 100
        X_CO2 = row["CO2 (Vol fr)"] / 100
        X_CO = row["CO (Vol fr)"] / 100

        delta_P = row["Pe (Pa)"]
        T_e = row["Te (K)"]

        return calculate_HRR(
            X_O2,
            X_CO2,
            X_CO,
            X_O2_initial,
            X_CO2_initial,
            delta_P,
            T_e,
            c_factor,
            e,
            area,
        )

    data["HRR (kW/m2)"] = data.apply(get_HRR, axis=1)

    # Calculate MFR by row

    def get_MFR(row):
        delta_P = row["Pe (Pa)"]
        T_e = row["Te (K)"]

        return calculate_MFR(c_factor, delta_P, T_e)

    data["MFR (kg/s)"] = data.apply(get_MFR, axis=1)

    return data

if __name__ == "__main__":
    parse_dir(INPUT_DIR)

    
