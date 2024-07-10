import pandas as pd
from pathlib import Path
import json
from datetime import datetime
import matplotlib.pyplot as plt
from graph import compare_FTT

# remove later
from pprint import pp

from processing import calculate_HRR, calculate_HRR_O2_only

INPUT_DIR = "./DATA/FTT"
OUTPUT_DIR = "./OUTPUT/FTT"
DEBUG = True


def parse_dir(input_dir):
    # read all CSV files in directory
    paths = Path(input_dir).glob("*.csv")

    # ignore the reduced data files (can be recalculated later from raw data)
    paths = filter(lambda x: not x.stem.endswith("red"), list(paths))

    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    # for each file, parse it
    for path in paths:
        try:
            parse_file(path)
        except Exception as e:
            print(f"Error parsing {path}: {e}")

    compare_FTT()


def parse_file(file_path):
    print(f"Parsing: {file_path}")

    # read in CSV file as pandas dataframe
    # note the csv files are encoded in cp1252 NOT utf-8...
    df = pd.read_csv(file_path, encoding="cp1252")

    metadata = parse_metadata(df)
    metadata_output_path = Path(OUTPUT_DIR) / f"{Path(file_path).stem}_metadata.json"

    with open(metadata_output_path, "w+") as f:
        json.dump(metadata, f, indent=4)

    data = parse_data(df)

    # calculate HRR
    data = process_data(
        data,
        c_factor=metadata["c_factor"],
        e=metadata["e_mj/kg"],
        area=metadata["surface_area_cm^2"],
        # change o2, co2, co delays to real numbers!
        o2_delay=metadata["o2_delay_time_s"],
        co2_delay=metadata["co2_delay_time_s"],
        co_delay=metadata["co_delay_time_s"],
        # rel_humidity=metadata["relative_humidity_%"],
        # T_a=metadata["ambient_temp"],
        # P_a=metadata["barometric_pressure_pa"],
    )

    if not DEBUG:
        data = data[
            "Time(s)",
            "Mass (g)",
            "O2 (%)",
            "CO2 (%)",
            "CO (%)",
        ]

    data_output_path = Path(OUTPUT_DIR) / f"{Path(file_path).stem}_data.csv"
    data.to_csv(data_output_path, index=False)

    # plot HRR
    # plt.figure(figsize=(8, 4))
    # plt.plot(data["HRR (kW/m2)"])
    # plt.plot(data["HRR (kW/m2)"].rolling(10, center=True).mean())
    # plt.tight_layout()
    # plt.show()


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

    metadata["pretest_comments"] = raw_metadata["Pre-test comments"]
    metadata["posttest_comments"] = raw_metadata["After-test comments"]

    metadata["grid"] = get_bool("Grid?")
    metadata["mounting_type"] = "Edge frame" if get_bool("Edge frame?") else None

    metadata["heat_flux_kw/m^2"] = get_number("Heat flux (kW/m²)")
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
    metadata["eot_time_s"] = get_number("User EOT time (s)")

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


def parse_data(df):
    def process_name(name):
        # remove parentheses, spaces, and convert to lowercase
        name = name.replace("(", "").replace(")", "").replace(" ", "_").lower()

        # replace unicode characters with their ASCII equivalents
        name = name.replace("°", "").replace("²", "^2").replace("³", "^3")

        return name

    # data is found in the remaining columns of the dataframe (column 3 onwards)
    data = df[df.columns[2:]]
    data.columns = [process_name(c) for c in data.columns]

    # we only care about certain columns, so only get that subset

    data = data[["time_s", "mass_g", "o2_%", "co2_%", "co_%", "dpt_pa", "stack_tc_k"]]
    data = data.rename(
        columns={
            "time_s": "Time (s)",
            "mass_g": "Mass (g)",
            "o2_%": "O2 (%)",
            "co2_%": "CO2 (%)",
            "co_%": "CO (%)",
            "dpt_pa": "DPT (Pa)",
            "stack_tc_k": "Stack TC (K)",
        }
    )

    return data


def process_data(
    data, c_factor, e, area, o2_delay, co2_delay, co_delay
):  # e is in mj/kg

    # shift columns up to account for O2, CO, CO2 analyzer time delay, and remove the rows at the end
    data["O2 (%)"] = data["O2 (%)"].shift(-o2_delay)

    data["CO2 (%)"] = data["CO2 (%)"].shift(-co2_delay)

    data["CO (%)"] = data["CO (%)"].shift(-co_delay)

    data.drop(data.tail(max(o2_delay, co2_delay, co_delay)).index, inplace=True)

    # convert area from cm^2 to m^2
    area = area / (100**2)

    # Calculate HRR by row

    def get_HRR(row):
        X_O2 = row["O2 (%)"] / 100
        X_O2_initial = data["O2 (%)"][:30].mean() / 100

        X_CO2 = row["CO2 (%)"] / 100
        X_CO2_initial = data["CO2 (%)"][:30].mean() / 100

        X_CO = row["CO (%)"] / 100
        X_CO_initial = data["CO (%)"][:30].mean() / 100

        delta_P = row["DPT (Pa)"]
        T_e = row["Stack TC (K)"]

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

        # return calculate_HRR_O2_only(
        #     X_O2,
        #     X_O2_initial,
        #     delta_P,
        #     T_e,
        #     c_factor,
        #     e,
        #     area,
        # )

    data["HRR (kW/m2)"] = data.apply(get_HRR, axis=1)

    return data


if __name__ == "__main__":
    parse_dir(INPUT_DIR)
    # parse_file("./DATA/FTT/19020031.csv")
