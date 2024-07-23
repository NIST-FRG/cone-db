import pandas as pd
from pathlib import Path
import json
from datetime import datetime
import matplotlib.pyplot as plt
from graph import compare_FTT
import scipy.signal as signal
import numpy as np
from dateutil import parser

from utils import calculate_HRR, calculate_MFR, colorize

INPUT_DIR = Path(r"./DATA/FTT")
OUTPUT_DIR = Path(r"./OUTPUT/FTT")
DEBUG = True


def parse_dir(input_dir):
    # read all CSV files in directory
    paths = Path(input_dir).glob("**/*.csv")

    # ignore the reduced data files (can be recalculated later from raw data)
    paths = filter(lambda x: not x.stem.endswith("red"), list(paths))
    # also remove scaled, stdev, or Output, etc. files since those are MIDAS format most likely
    paths = list(
        filter(
            lambda x: not x.stem.endswith(
                ("scaled", "stdev", "Output", "raw", "Post", "PrelimReport")
            ),
            list(paths),
        )
    )

    total_files = len(paths)

    print(colorize(f"Found {len(paths)} files to parse", "purple"))

    files_parsed = 0
    files_parsed_successfully = 0

    # for each file, parse it
    for path in paths:
        if files_parsed % 20 == 0 and files_parsed != 0:
            print(
                colorize(
                    f"Files parsed successfully: {files_parsed_successfully}/{files_parsed} ({(files_parsed_successfully/files_parsed) * 100}%), total files: {total_files}",
                    "blue",
                )
            )

        try:
            files_parsed += 1
            parse_file(path)
        except Exception as e:
            print(colorize(f" - Error parsing {path}: {e}", "red"))
            continue
        print(colorize(" - Parsed successfully", "green"))
        files_parsed_successfully += 1

    print(
        colorize(
            f"COMPLETE: Files parsed successfully: {files_parsed_successfully}/{files_parsed} ({(files_parsed_successfully/files_parsed) * 100}%), total files: {total_files}",
            "blue",
        )
    )


def parse_file(file_path):
    print(f"Parsing: {file_path}")

    # read in CSV file as pandas dataframe
    # note the csv files are encoded in cp1252 NOT utf-8...
    df = pd.read_csv(file_path, encoding="cp1252")

    # if the second row is blank, drop it
    if df.iloc[1].isnull().all():
        df = df.drop(1)

    metadata = parse_metadata(df)

    data = parse_data(df, metadata)

    # If there's less than 20 data points, just skip the file
    if len(data) < 20:
        print(colorize(f"Skipping {file_path} because it has less than 20 seconds of data", "yellow"))
        return

    test_year = parser.parse(metadata["date"]).year

    # Determine output path
    Path(OUTPUT_DIR / str(test_year)).mkdir(parents=True, exist_ok=True)

    data_output_path = Path(OUTPUT_DIR) / str(test_year) /f"{Path(file_path).stem}.csv"
    metadata_output_path = Path(OUTPUT_DIR) / str(test_year) / f"{Path(file_path).stem}.json"

    with open(metadata_output_path, "w+") as f:
        json.dump(metadata, f, indent=4)

    data.to_csv(data_output_path, index=False)


def parse_metadata(df):
    # metadata is found in the first two columns of the dataframe
    raw_metadata = df[df.columns[:2]].dropna(how="all")

    # get the tranpose of the dataframe (swaps rows & columns)
    raw_metadata = raw_metadata.T

    # change headers to be the first row of the dataframe
    new_header = raw_metadata.iloc[0]
    raw_metadata = raw_metadata[1:]
    raw_metadata.columns = new_header

    raw_metadata = raw_metadata.to_dict(orient="list")
    raw_metadata = {k: v[0] if len(v) > 0 else None for k, v in raw_metadata.items()}

    # pp(raw_metadata)

    metadata = {}

    # Process date

    def get_number(key):
        try:
            return float(raw_metadata[key])
        except:
            return None

    def get_bool(key):
        if raw_metadata[key] == "Yes":
            return True
        elif raw_metadata[key] == "No":
            return False
        else:
            return None

    # Date parsing
    raw_date = raw_metadata["Date of test"]
    raw_time = raw_metadata["Time of test"]

    date = datetime.strptime(f"{raw_date} {raw_time}", "%d/%m/%Y %H:%M")

    metadata["date"] = date.isoformat()

    # Process other metadata
    metadata["laboratory"] = raw_metadata["Laboratory name"]
    metadata["operator"] = raw_metadata["Operator"]
    metadata["report_name"] = raw_metadata["Report name"]

    # metadata["pretest_comments"] = raw_metadata["Pre-test comments"]
    # metadata["posttest_comments"] = raw_metadata["After-test comments"]
    comments = ""
    if raw_metadata['Pre-test comments'] == raw_metadata["Pre-test comments"]:
        comments += f"Pre-test: {raw_metadata['Pre-test comments']}\n"
    if raw_metadata["After-test comments"] == raw_metadata["After-test comments"]:
        comments += f"Post-test: {raw_metadata["After-test comments"]}\n"
    
    metadata["comments"] = comments

    metadata["grid"] = get_bool("Grid?")
    metadata["mounting_type"] = "Edge frame" if get_bool("Edge frame?") else None

    metadata["heat_flux_kW/m^2"] = get_number("Heat flux (kW/m²)")
    metadata["separation_mm"] = get_number("Separation (mm)")

    metadata["manufacturer"] = raw_metadata["Manufacturer"]
    metadata["material_name"] = raw_metadata["Material name/ID"]
    metadata["specimen_description"] = raw_metadata["Sample description"]
    metadata["specimen_number"] = raw_metadata["Specimen number"]
    metadata["specimen_prep"] = raw_metadata["Additional specimen preparation details"]
    metadata["sponsor"] = raw_metadata["Sponsor"]

    metadata["thickness_mm"] = get_number("Thickness (mm)")
    metadata["surface_area_cm^2"] = get_number("Surface area (cm²)")

    metadata["time_to_ignition_s"] = get_number("Time to ignition (s)")
    metadata["time_to_flameout_s"] = get_number("Time to flameout (s)")
    metadata["test_start_time_s"] = get_number("Test start time (s)")
    metadata["test_end_time_s"] = get_number("User EOT time (s)")

    metadata["mlr_eot_mass_g/m^2"] = get_number("MLR EOT mass (g/m²)")
    metadata["eot_criterion"] = get_number("End of test criterion")

    metadata["e_mj/kg"] = get_number("E (MJ/kg)")
    metadata["od_correction_factor"] = get_number("OD correction factor")

    metadata["initial_mass_g"] = get_number("Initial mass (g)")

    metadata["substrate"] = raw_metadata["Substrate"]
    metadata["non_scrubbed"] = get_bool("Non-scrubbed?")

    metadata["orientation"] = raw_metadata["Orientation"]

    metadata["c_factor"] = get_number("C-factor (SI units)")
    metadata["duct_diameter_m"] = raw_metadata["Duct diameter (m)"]

    metadata["o2_delay_time_s"] = get_number("O2 delay time (s)")
    metadata["co2_delay_time_s"] = get_number("CO2 delay time (s)")
    metadata["co_delay_time_s"] = get_number("CO delay time (s)")

    metadata["ambient_temp"] = get_number("Ambient temperature (°C)")
    metadata["barometric_pressure_pa"] = get_number("Barometric pressure (Pa)")
    metadata["relative_humidity_%"] = get_number("Relative humidity (%)")

    metadata["co_co2_data_collected"] = get_bool("CO/CO2 data collected?")
    metadata["mass_data_collected"] = get_bool("Mass data collected?")
    metadata["smoke_data_collected"] = get_bool("Smoke data collected?")
    metadata["soot_mass_data_collected"] = get_bool("Soot mass data collected?")

    metadata["soot_mass_g"] = get_number("Soot mass (g)")
    metadata["soot_mass_ratio"] = get_number("Soot mass ratio (1:x)")

    metadata["e_mj/kg"] = get_number("E (MJ/kg)")

    # TODO: would it be better to convert this to hertz
    metadata["sampling_interval_s"] = get_number("Sampling interval (s)")

    # replace all NaN values with None (which turns into null when serialized) to fit JSON spec
    metadata = {k: v if v == v else None for k, v in metadata.items()}

    # pp(metadata)

    return metadata


def parse_data(df, metadata):
    def process_name(name):
        # remove parentheses, spaces, and convert to lowercase
        name = name.replace("(", "").replace(")", "").replace(" ", "_").lower()

        # replace unicode characters with their ASCII equivalents
        name = name.replace("°", "").replace("²", "^2").replace("³", "^3")

        return name

    # data is found in the remaining columns of the dataframe (column 3 onwards)
    data = df[df.columns[2:]]
    # get rid of some of the special characters in the column names to make it easier to work with
    data.columns = [process_name(c) for c in data.columns]

    # convert O2, CO2, and CO into vol fr
    data.loc[:, "o2_%"] /= 100
    data.loc[:, "co2_%"] /= 100
    data.loc[:, "co_%"] /= 100

    # we only care about certain columns, so only get that subset
    data = data.rename(
        columns={
            "time_s": "Time (s)",
            "mass_g": "Mass (g)",
            "o2_%": "O2 (Vol fr)",
            "co2_%": "CO2 (Vol fr)",
            "co_%": "CO (Vol fr)",
            "dpt_pa": "DPT (Pa)",
            "stack_tc_k": "Te (K)",
        }
    )

    # If the Time (s) column has increments besides 1, raise an error
    if data["Time (s)"].diff().max() > 1:
        raise Exception("Time increments are not 1 second")

    # get metadata required for HRR calculations

    data = process_data(data, metadata)

    # keep only the columns we need
    data = data[
        [
            "Time (s)",
            "Mass (g)",
            "O2 (Vol fr)",
            "CO2 (Vol fr)",
            "CO (Vol fr)",
            "HRR (kW/m2)",
            "MFR (kg/s)",
        ]
    ]

    return data


def process_data(data, metadata):

    start_time = int(metadata.get("test_start_time_s", 0))
    o2_delay = int(metadata["o2_delay_time_s"] or 0)
    co2_delay = int(metadata["co2_delay_time_s"] or 0)
    co_delay = int(metadata["co_delay_time_s"] or 0)
    area = metadata["surface_area_cm^2"] or 100  # cm^2
    c_factor = metadata["c_factor"]
    e = metadata["e_mj/kg"]

    # convert area from cm^2 to m^2
    area = area / (100**2)

    # if start-time is not defined, just use the first 30 secs for baseline
    baseline_end = int(start_time if start_time > 0 else 30)

    # calculate baseline values by using the data up to test start time
    X_O2_initial = data["O2 (Vol fr)"][:baseline_end].mean()  # / 100
    X_CO2_initial = data["CO2 (Vol fr)"][:baseline_end].mean()  # / 100
    X_CO_initial = data["CO (Vol fr)"][:baseline_end].mean()  # / 100

    # shift entire dataframe up to start time
    data = data.shift(-start_time)
    data.drop(data.tail(start_time).index, inplace=True)

    data["Time (s)"] = data["Time (s)"] - start_time

    # shift certain columns up to account for O2, CO, CO2 analyzer time delay, and remove the rows at the end

    data["O2 (Vol fr)"] = data["O2 (Vol fr)"].shift(-o2_delay)
    data["CO2 (Vol fr)"] = data["CO2 (Vol fr)"].shift(-co2_delay)
    data["CO (Vol fr)"] = data["CO (Vol fr)"].shift(-co_delay)

    data.drop(data.tail(max(o2_delay, co_delay, co2_delay)).index, inplace=True)

    # Calculate HRR by row

    def get_HRR(row):
        X_O2 = row["O2 (Vol fr)"]  # / 100
        X_CO2 = row["CO2 (Vol fr)"]  # / 100
        X_CO = row["CO (Vol fr)"]  # / 100

        delta_P = row["DPT (Pa)"]
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
        delta_P = row["DPT (Pa)"]
        T_e = row["Te (K)"]

        return calculate_MFR(c_factor, delta_P, T_e)

    data["MFR (kg/s)"] = data.apply(get_MFR, axis=1)

    return data


if __name__ == "__main__":
    parse_dir(INPUT_DIR)
    # parse_file("./DATA/FTT/24030001.csv")
