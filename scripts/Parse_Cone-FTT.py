import pandas as pd
from pathlib import Path
import json
import sys
from datetime import datetime
from dateutil import parser
import numpy as np
import sys
from utils import calculate_HRR, calculate_MFR, colorize

# first argument is the input directory, 2nd argument is the output directory
args = sys.argv[1:]
if len(args) > 2:
    print("""Too many arguments
          Usage: python parse-FTT.py <input_dir> <output_dir>
          Leave empty to use defaults.""")
    sys.exit()


#Path Handling: Relative to this script's location
SCRIPT_DIR = Path(__file__).resolve().parent         # .../coneDB/scripts
PROJECT_ROOT = SCRIPT_DIR.parent             # .../coneDB 

INPUT_DIR1 = Path(r"\\firedata\FLAMMABILITY_DATA\DATA\Cone\FTT-White\Data")
OUTPUT_DIR1 = PROJECT_ROOT / "Exp-Data_Parsed" / "FTT-White"
META_DIR1 = PROJECT_ROOT / "Metadata" / "Parsed" / "FTT-White"
LOG_FILE1 = PROJECT_ROOT/ "parse_FTT-White_log.json"

INPUT_DIR2 = Path(r"\\firedata\FLAMMABILITY_DATA\DATA\Cone\FTT-Black\Data")
OUTPUT_DIR2 = PROJECT_ROOT / "Exp-Data_Parsed" / "FTT-Black"
META_DIR2 = PROJECT_ROOT / "Metadata" / "Parsed" / "FTT-Black"
LOG_FILE2 = PROJECT_ROOT / "parse_FTT-Black_log.json"

if len(args) == 2:
    INPUT_DIR = Path(args[0])
    OUTPUT_DIR = Path(args[1])

#region parse_dir
def parse_dir(input_dir):
    OUTPUT_DIR = None
    META_DIR = None
    LOG_FILE = None
    color = None
    if "White" in str(input_dir):
        OUTPUT_DIR = OUTPUT_DIR1
        META_DIR = META_DIR1
        LOG_FILE = LOG_FILE1
        color = "White"
    elif "Black" in str(input_dir):
        OUTPUT_DIR = OUTPUT_DIR2
        META_DIR = META_DIR2
        LOG_FILE = LOG_FILE2
        color = 'Black'
    logfile = {}
    with open(LOG_FILE, "w", encoding="utf-8") as f:
        f.write(json.dumps(logfile, indent=4))
    print(f"✅ parse_FTT-{color}.json created.")
    
    # read all CSV files in directory
    paths = Path(input_dir).glob("**/*.CSV")
    

    # ignore the reduced data files (can be recalculated later from raw data)
    paths = filter(lambda x: not x.stem.endswith("_red"), list(paths))
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
            parse_file(path, OUTPUT_DIR, META_DIR)
        except Exception as e:
            with open(LOG_FILE, "r", encoding="utf-8") as w:  
                logfile = json.load(w)
            logfile.update({
                    str(path) : str(e)
                })
            with open(LOG_FILE, "w", encoding="utf-8") as f:
	            f.write(json.dumps(logfile, indent=4))
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

#region parse_file
def parse_file(file_path, output, meta):
    print(f"Parsing: {file_path}")

    # read in CSV file as pandas dataframe
    # note the csv files are encoded in cp1252 NOT utf-8...
    df = pd.read_csv(file_path, encoding="cp1252")

    # if the second row is blank, drop it
    if df.iloc[1].isnull().all():
        df = df.drop(1)

    # TODO
    df = df.dropna(how="all")

    metadata = parse_metadata(df,file_path, meta)

    data = parse_data(df, metadata)

    # If there's less than 20 data points, just skip the file
    if len(data) < 20:
        print(colorize(f"Skipping {file_path} because it has less than 20 seconds of data", "yellow"))
        return

    test_year = parser.parse(metadata["Test Date"]).year

    # Determine output path
    Path(output/ str(test_year)).mkdir(parents=True, exist_ok=True)
    Path(meta/ str(test_year)).mkdir(parents=True, exist_ok=True)
    data_output_path = Path(output) / str(test_year) /f"{Path(file_path).stem}.csv"
    metadata_output_path = Path(meta) / str(test_year) / f"{Path(file_path).stem}.json"

    with open(metadata_output_path, "w+") as f:
        json.dump(metadata, f, indent=4)

    data.to_csv(data_output_path, index=False)

#region parse_metadata
def parse_metadata(df,file_path, meta):
    # metadata is found in the first two columns of the dataframe
    raw_metadata = df[df.columns[:2]].dropna(how="all")

    # get the tranpose of the dataframe (swaps rows & columns)
    raw_metadata = raw_metadata.T

    # change headers to be the first row of the dataframe
    new_header = raw_metadata.iloc[0]
    raw_metadata = raw_metadata[1:]
    raw_metadata.columns = new_header

    # convert metadata dataframe to be a dictionary
    raw_metadata = raw_metadata.to_dict(orient="list")
    raw_metadata = {k: v[0] if len(v) > 0 else None for k, v in raw_metadata.items()}

    metadata = {}

    # helper functions to extract values from the metadata dictionary

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

    #region metadata properties

    # Date parsing
    raw_date = raw_metadata["Date of test"]
    raw_time = raw_metadata["Time of test"]
    
    # Remove seconds from time string
    raw_time_rounded = raw_time[:5]  # Get only the first 5 characters (HH:MM)

    # Combine date and time into one string
    datetime_string = f"{raw_date} {raw_time_rounded}"

    # Parse without seconds
    date = datetime.strptime(datetime_string, "%d/%m/%Y %H:%M")

    metadata["Test Date"] = date.isoformat()

    # Process other metadata
    metadata["Insitituion"] = raw_metadata["Laboratory name"]
    metadata["Operator"] = raw_metadata["Operator"]
    metadata["Report Name"] = raw_metadata["Report name"]

    # pre-test and post-test comments are combined into one "comments" field
    comments = ""
    if raw_metadata['Pre-test comments'] == raw_metadata["Pre-test comments"]:
        comments += f"Pre-test: {raw_metadata['Pre-test comments']}\n"
    if raw_metadata["After-test comments"] == raw_metadata["After-test comments"]:
        comments += f"Post-test: {raw_metadata["After-test comments"]}\n"
    metadata["Comments"] = comments

    metadata["Grid"] = get_bool("Grid?")
    metadata["Mounting_type"] = "Edge frame" if get_bool("Edge frame?") else None

    metadata["Heat flux (kW/m2)"] = get_number("Heat flux (kW/m²)")
    metadata["Separation (mm)"] = get_number("Separation (mm)")

    metadata["Manufacturer"] = raw_metadata["Manufacturer"]
    metadata["Material Name"] = raw_metadata["Material name/ID"]
    metadata["Sample Description"] = raw_metadata["Sample description"]
    metadata["Specimen Number"] = raw_metadata["Specimen number"]
    metadata["Specimen Prep"] = raw_metadata["Additional specimen preparation details"]
    metadata["Sponsor"] = raw_metadata["Sponsor"]

    metadata["Thickness (mm)"] = get_number("Thickness (mm)")
    metadata["Surface Area (m2)"] = get_number("Surface area (cm²)") * 0.0001

    metadata["Time to Ignition (s)"] = get_number("Time to ignition (s)")
    metadata["Time to Flameout (s)"] = get_number("Time to flameout (s)")
    metadata["Test Start Time (s)"] = get_number("Test start time (s)")
    metadata["Test End Time (s)"] = get_number("User EOT time (s)")

    metadata["MLR EOT Mass (g/m2)"] = get_number("MLR EOT mass (g/m²)")
    metadata["End of test criterion"] = get_number("End of test criterion")

    metadata["E (MJ/kg)"] = get_number("E (MJ/kg)")
    metadata["OD Correction Factor"] = get_number("OD correction factor")

    metadata["Sample Mass (g)"] = get_number("Initial mass (g)")

    metadata["Substrate"] = raw_metadata["Substrate"]
    metadata["Non-scrubbed"] = get_bool("Non-scrubbed?")

    metadata["Orientation"] = raw_metadata["Orientation"]

    metadata["C Factor"] = get_number("C-factor (SI units)")
    metadata["Duct Diameter (m)"] = raw_metadata["Duct diameter (m)"]

    metadata["O2 Delay Time (s)"] = get_number("O2 delay time (s)")
    metadata["CO2 Delay Time (s)"] = get_number("CO2 delay time (s)")
    metadata["CO Delay Time (s)"] = get_number("CO delay time (s)")

    metadata["Ambient Temperature (°C)"] = get_number("Ambient temperature (°C)")
    metadata["Barometric Pressure (Pa)"] = get_number("Barometric pressure (Pa)")
    metadata["Relative Humidity (%)"] = get_number("Relative humidity (%)")

    metadata["Sampling Interval (s)"] = get_number("Sampling interval (s)")

    expected_keys = [
        "Institution",
        "Heat Flux (kW/m2)",
        "Material Name",
        "Orientation",
        "C Factor",
        "Sample Mass (g)",
        "Residual Mass (g)",
        "Surface Area (m2)",
        "Soot Average (g/g)",
        "Mass Consumed",
        "Conversion Factor",
        "Time to Ignition (s)",
        "Peak Heat Release Rate (kW/m2)",
        "Peak Mass Loss Rate (g/s-m2)",
        "Specimen Number",
        "Test Date",
        "Residue Yield (g/g)"
    ]
    cone = "White" if "White" in str(meta) else "Black"
    for key in expected_keys:
        metadata.setdefault(key, None)
    metadata['Original Testname'] = file_path.stem
    metadata ['Testname'] = None
    metadata['Instrument'] = f"{cone} FTT Cone Calorimeter"
    metadata['Autoprocessed'] = None
    metadata['Preparsed'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    metadata['Parsed'] = None
    metadata['SmURF'] = None
    metadata['Bad Data'] = None
    metadata["Auto Prepared"] = None
    metadata["Manually Prepared"] = None
    metadata["Manually Reviewed Series"] = None
    metadata['Pass Review'] = None
    metadata["Published"] = None
    metadata["Original Source"] = f"FTT-{cone}"
    metadata['Data Corrections'] =[]



    # replace all NaN values with None (which turns into null when serialized) to fit JSON spec (and get rid of red underlines)
    metadata = {k: v if v == v else None for k, v in metadata.items()}

    return metadata

#region parse_data
def parse_data(df, metadata):

    # data is found in the remaining columns of the dataframe (column 3 onwards)
    data = df[df.columns[2:]]

    # convert O2, CO2, and CO into vol fr
    data.loc[:, "O2 (%)"] /= 100
    data.loc[:, "CO2 (%)"] /= 100
    data.loc[:, "CO (%)"] /= 100

    # rename columns for consistency
    data = data.rename(
        columns={
            "time (s)": "Time (s)",
            "O2 (%)": "O2 (Vol fr)",
            "CO2 (%)": "CO2 (Vol fr)",
            "CO (%)": "CO (Vol fr)",
            "Stack TC (K)": "Te (K)",
        }
    )

    # If the Time (s) column has increments besides 1, raise an error
    if data["Time (s)"].diff().max() > 1:
        raise Exception("Time increments are not 1 second")
    # get metadata required for HRR calculations
    data = process_data(data, metadata)
    # selected columns to keep
    data = data[
        [
            "Time (s)",
            "Mass (g)",
            "HRR (kW)",
            "O2 (Vol fr)",
            "CO2 (Vol fr)",
            "CO (Vol fr)",
            "MFR (kg/s)",
            "K Smoke (1/m)"
        ]
    ]

    return data

#region process_data
def process_data(data, metadata):

    # test parameters used for calculations
    start_time = int(metadata.get("Test Start Time (s)", 0))
    o2_delay = int(metadata["O2 Delay Time (s)"] or 0)
    co2_delay = int(metadata["CO2 Delay Time (s)"] or 0)
    co_delay = int(metadata["CO Delay Time (s)"] or 0)
    area = metadata["Surface Area (m2)"] or .0001  # cm2
    c_factor = metadata["C Factor"]
    e = metadata["E (MJ/kg)"]
    duct_length = float(metadata["Duct Diameter (m)"]) or 0.114 

    #region delay, baselines

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

    # shift certain columns up to account for O2, CO, CO2 analyzer time delay, and remove the rows at the end from the other signals
    data.drop(data.tail(max(o2_delay, co_delay, co2_delay)).index, inplace=True)

    #region calc. HRR & MFR

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

    data["HRRPUA (kW/m2)"] = data.apply(get_HRR, axis=1)

    # Calculate MFR by row

    def get_MFR(row):
        delta_P = row["DPT (Pa)"]
        T_e = row["Te (K)"]

        return calculate_MFR(c_factor, delta_P, T_e)

    data["MFR (kg/s)"] = data.apply(get_MFR, axis=1)
    
    data["K Smoke (1/m)"] = (1/duct_length) * np.log(data["PDC (-)"]/data["PDM (-)"])
    data["HRR (kW)"] = data["HRRPUA (kW/m2)"] * area

    return data


if __name__ == "__main__":
    parse_dir(INPUT_DIR1)
    parse_dir(INPUT_DIR2)
    # parse_file("./DATA/FTT/24030001.csv")
