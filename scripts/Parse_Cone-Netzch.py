
import pandas as pd
from io import StringIO
from pathlib import Path
import json
import sys
import os
from datetime import datetime
from dateutil import parser
import matplotlib.pyplot as plt
import numpy as np
import sys
from scipy.signal import savgol_filter
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

INPUT_DIR = Path(r"C:\Users\rtg4\Desktop\Netzsch_Cone")
OUTPUT_DIR = PROJECT_ROOT / "Exp-Data_Parsed" / "Netzsch"
META_DIR = PROJECT_ROOT / "Metadata" / "Parsed" / "Netzsch"
LOG_FILE = PROJECT_ROOT/ "parse_Netzsch_log.json"



def parse_dir(input_dir):
    logfile = {}
    with open(LOG_FILE, "w", encoding="utf-8") as f:
        f.write(json.dumps(logfile, indent=4))
    print(f"✅ parse_Netzsch_log.json created.")
    
    # read all CSV files in directory
    paths = Path(input_dir).glob("**/*.CSV")
    paths = filter(lambda x: not x.stem.endswith("_corrected"), list(paths))
    paths = list(paths)
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



def parse_file(file_path, output, meta):
    print(f"Parsing: {Path(file_path).stem}")

    with open(file_path, "r", encoding="latin1") as f:
        raw_lines = [line.rstrip("\n") for line in f if line.strip()]

    # Determine correct column count from header
    header = raw_lines[0]
    n_cols = header.count(";") + 1

    clean_lines = [header]

    for line in raw_lines[1:]:
        parts = line.split(";")

        if len(parts) > n_cols:
            parts = parts[:n_cols]           # truncate extra fields
        elif len(parts) < n_cols:
            parts.extend([""] * (n_cols - len(parts)))  # pad missing fields

        clean_lines.append(";".join(parts))

    # Load into pandas
    df = pd.read_csv(
        StringIO("\n".join(clean_lines)),
        sep=";",
        decimal=",",
        engine="python"
    )
    #input_path = Path(file_path)
    #output_path = input_path.parent / f"{input_path.stem}_corrected.csv"

    # Save as American-style CSV
    #df.to_csv(output_path, index=False)

        # if the second row is blank, drop it
    df = df.drop(0).reset_index(drop=True)

    df = df.dropna(how="all")
    metadata = parse_metadata(df,file_path, meta)
    RawMod = os.path.getmtime(file_path)
    RawMod = datetime.fromtimestamp(RawMod).strftime('%Y-%m-%d %H:%M:%S')
    data, metadata = parse_data(df, metadata)

    #Final mass into the metadata : MAY NEED TO ADJUST THIS SINCE SOME TESTS GO NEGATIVE
    metadata["Residual Mass (g)"] = data['Mass (g)'].iloc[-1]


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
    if os.path.exists(metadata_output_path):
        with open(metadata_output_path, "r") as f:
            existing_meta = json.load(f)
        if existing_meta["Parsed"] > RawMod:
            print(colorize(f"Files for already exist and are up to date.", 'yellow'))
            return

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
    expected_keys = [
    "Material ID",
    "Material Name",
    "Sample Mass (g)",
    "Residual Mass (g)",
    "Specimen Number",
    "Original Testname",
    "Testname",
    "Thickness (mm)",
    "Sample Description",
    "Specimen Prep",
    "Instrument",
    "Test Date",
    "Test Time",
    "Operator",
    "Director",
    "Sponsor",
    "Institution",
    "Report Name",
    "Original Source",
    "Parsed",
    "Auto Prepared",
    "Manually Prepared",
    "SmURF", 
    "Bad Data",
    "Autoprocessed",
    "Manually Reviewed Series",
    "Pass Review",
    "Published",
    "Heat Flux (kW/m2)",
    "Orientation",
    "C Factor",
    "Surface Area (m2)",
    "Grid",
    "Edge Frame",
    "Separation (mm)",
    "Test Start Time (s)",
    "Test End Time (s)",
    "MLR EOT Mass (g/m2)",
    "End of test criterion",
    "Heat of Combustion O2 (MJ/kg)",
    "OD Correction Factor",
    "Substrate",
    "Non-scrubbed",
    "Duct Diameter (m)",
    "O2 Delay Time (s)",
    "CO2 Delay Time (s)",
    "CO Delay Time (s)",
    "Ambient Temperature (°C)",
    "Barometric Pressure (Pa)",
    "Relative Humidity (%)",
    "X_O2 Initial", "X_CO2 Initial", 'X_CO Initial',
    't_ignition (s)', 't_ignition Outlier',
    'm_ignition (g)', 'm_ignition Outlier',
    'Residue Yield (%)', 'Residue Yield Outlier',
    'Heat Release Rate Outlier',
    'Average HRRPUA 60s (kW/m2)','Average HRRPUA 60s Outlier',
    'Average HRRPUA 180s (kW/m2)','Average HRRPUA 180s Outlier',
    'Average HRRPUA 300s (kW/m2)', 'Average HRRPUA 300s Outlier',
    'Steady Burning MLRPUA (g/s-m2)', 'Steady Burning MLRPUA Outlier',
    'Peak MLRPUA (g/s-m2)','Peak MLRPUA Outlier',
    'Steady Burning HRRPUA (kW/m2)', 'Steady Burning HRRPUA Outlier',
    'Peak HRRPUA (kW/m2)', 'Peak MLRPUA Outlier',
    'Total Heat Release (MJ/m2)', 'Total Heat Release Outlier',
    'Average HoC (MJ/kg)', 'Average HoC Outlier',
    'Average Specific Extinction Area (m2/kg)', 'Average Specific Extinction Area Outlier',
    'Smoke Production Pre-ignition (m2/m2)','Smoke Production Pre-ignition Outlier',
    'Smoke Production Post-ignition (m2/m2)','Smoke Production Post-ignition Outlier',
    'Smoke Production Total (m2/m2)','Smoke Production Total Outlier',
    'Y_Soot (g/g)', 'Y_Soot Outlier',
    'Y_CO2 (g/g)', 'Y_CO2 Outlier',
    'Y_CO (g/g)', 'Y_CO Outlier',
    'Fire Growth Potential (m2/J)', 'Fire Growth Potential Outlier',
    'Ignition Energy (MJ/m2)', 'Ignition Energy Outlier',
    "t_flameout (s)","t_flameout outlier",
    'Comments', 'Data Corrections'
        ]

    for key in expected_keys:
        metadata.setdefault(key, None)


    # helper functions to extract values from the metadata dictionary
    def get_number(key):
        try:
            return float(raw_metadata[key])
        except:
            return None

    def get_bool(key):
        if raw_metadata[key] == "Yes" or raw_metadata[key] == "Y":
            return True
        elif raw_metadata[key] == "No" or raw_metadata[key] == "N":
            return False
        else:
            return None

    #region metadata properties

    # Date parsing
    raw_date = raw_metadata["Date of test"]
    raw_time = raw_metadata["Time of test"]

    # Remove seconds from time string
    raw_time_rounded = raw_time[:5]  # e.g., "16:23:45" -> "16:23"

    # Parse date
    test_date = datetime.strptime(raw_date, '%d.%m.%Y').date()
    # Parse time (ignoring seconds)
    test_time = datetime.strptime(raw_time_rounded, "%H:%M").time()

    # Assign to metadata
    metadata["Test Date"] = test_date.isoformat()  # e.g., "2024-06-07"
    metadata["Test Time"] = test_time.strftime("%H:%M")  # e.g., "16:23"

    # Process other metadata
    metadata["Institution"] = raw_metadata["Laboratory name"]
    metadata["Operator"] = raw_metadata["Operator"]
    metadata["Report Name"] = raw_metadata["Report name"]

    # pre-test and post-test comments are combined into one "comments" field
    comments = []
    if raw_metadata.get('Pre-test comments'):
        comments.append(f"Pre-test: {raw_metadata['Pre-test comments']}")
    if raw_metadata.get('After test comments'):
        comments.append(f"Post-test: {raw_metadata['After test comments']}")
    metadata["Comments"] = comments

    metadata["Grid"] = get_bool("Grid?")
    metadata["Edge Frame"] = get_bool("Edge frame?") 

    metadata["Heat Flux (kW/m2)"] = get_number("Heat flux (kW/m²)")
    metadata["Separation (mm)"] = get_number("Separation (mm)")
    
    metadata["Material Name"] = raw_metadata["Material name/ID"]
    metadata["Sample Description"] = raw_metadata["Sample description"]
    metadata["Specimen Number"] = raw_metadata["Specimen number"]
    metadata["Specimen Prep"] = raw_metadata["Additional preparation details"]
    metadata["Sponsor"] = raw_metadata["Sponsor"]

    metadata["Thickness (mm)"] = get_number("Thickness (mm)")
    metadata["Surface Area (m2)"] = get_number("Surface area (cm²)") * 0.0001
    metadata["t_ignition (s)"] = get_number("Time to ignition (s)")
    metadata["t_flameout (s)"] = get_number("Time to flameout (s)")
    metadata["Test Start Time (s)"] = get_number("Test start time (s)")
    metadata["Test End Time (s)"] = get_number("User EOT time (s)")

    metadata["MLR EOT Mass (g/m2)"] = get_number("MLR EOT mass (g/m²)")
    metadata["End of test criterion"] = get_number("End of test criterion")

    metadata["Heat of Combustion O2 (MJ/kg)"] = get_number("E (MJ/kg)")
    metadata["OD Correction Factor"] = get_number("OD correction factor")

    metadata["Sample Mass (g)"] = get_number("Initial mass (g)")

    metadata["Substrate"] = raw_metadata["Substrate"]
    metadata["Non-scrubbed"] = get_bool("Non-scrubbed?")

    metadata["Orientation"] = raw_metadata["Orientation"]
    if metadata["Orientation"].lower() == "hor" or metadata["Orientation"].lower() == "horiz" or metadata["Orientation"].lower() == "h":
        metadata["Orientation"] = "Horizontal"
    elif metadata["Orientation"].lower() == "vert" or metadata["Orientation"].lower() == "ver" or metadata["Orientation"].lower() == "v":
        metadata["Orientation"] = "Vertical"

    metadata["C Factor"] = get_number("C-factor (SI units)")
    metadata["Duct Diameter (m)"] = get_number("Duct diameter (m)")

    metadata["O2 Delay Time (s)"] = get_number("O2 delay time (s)")
    metadata["CO2 Delay Time (s)"] = get_number("CO2 delay time (s)")
    metadata["CO Delay Time (s)"] = get_number("CO delay time (s)")

    metadata["Ambient Temperature (°C)"] = get_number("Ambient temperature (°C)")
    metadata["Barometric Pressure (Pa)"] = get_number("Barometric pressure (Pa)")
    metadata["Relative Humidity (%)"] = get_number("Relative humidity (%)")
    if metadata['Relative Humidity (%)'] is None:#TYPO IN NETSZCH RAW FILE
        metadata['Relative Humidity (%)'] = get_number("Relative hunidity (%)")


    metadata['Original Testname'] = file_path.stem
    metadata['Instrument'] = f"Netzsch Cone Calorimeter"
    metadata['Parsed'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    metadata["Original Source"] = f"Netzsch"
    metadata['Data Corrections'] =[]

    # replace all NaN values with None (which turns into null when serialized) to fit JSON spec (and get rid of red underlines)
    metadata = {k: v if v == v else None for k, v in metadata.items()}
    
    return metadata


#region parse_data
def parse_data(df, metadata):

    # data is found in the remaining columns of the dataframe (column 3 onwards)
    data = df[df.columns[2:]]
    mass_shift = data.loc[0, "Mass (g)"] - metadata["Sample Mass (g)"]
    data.loc[:, "Mass (g)"] = data.loc[:, "Mass (g)"] - mass_shift
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
            "Smoke TC (K)": "T Duct (K)"
        }
    )
    if "Te (K)" not in data.columns:#TYPO IN NETZSCH FILE
        data = data.rename(columns={
            "StackTC (K)": "Te (K)"
        })

    # If the Time (s) column has increments besides 1, raise an error
    if data["Time (s)"].diff().max() > 1:
        raise Exception("Time increments are not 1 second")
    # get metadata required for HRR calculations
    data, metadata = process_data(data, metadata)
    # selected columns to keep
    data = data[
       [
           "Time (s)",
            "Mass (g)",
            "HRR (kW)",
            "MFR (kg/s)",
            "T Duct (K)",
            "O2 (Vol fr)",
            "CO2 (Vol fr)",
            "CO (Vol fr)",
            "K Smoke (1/m)",
        ]
    ]

    return data, metadata

#region process_data
def process_data(data, metadata):
    # test parameters used for calculations
    start_time = int(metadata.get("Test Start Time (s)", -1))
    if start_time == -1: #Find start of test where mass not stable
        delta = data["Mass (g)"].diff().abs()
        test_start_index = delta[delta > 1e-3].index[0] 
        start_time = data.loc[test_start_index, "Time (s)"]
    
    # calculate initial values by using the data up to test start time
    X_O2_initial = data["O2 (Vol fr)"][:start_time].mean() 
    X_CO2_initial = data["CO2 (Vol fr)"][:start_time].mean()  
    X_CO_initial = data["CO (Vol fr)"][:start_time].mean()  
    
    
    o2_delay = int(metadata["O2 Delay Time (s)"] or 0)
    co2_delay = int(metadata["CO2 Delay Time (s)"] or 0)
    co_delay = int(metadata["CO Delay Time (s)"] or 0)
    area = metadata["Surface Area (m2)"] or .0001 
    c_factor = metadata["C Factor"]
    e = metadata["Heat of Combustion O2 (MJ/kg)"]
 
    duct_length = float(metadata["Duct Diameter (m)"]) or 0.114 
    amb_temp = float(metadata["Ambient Temperature (°C)"])

    rel_humid = float(metadata['Relative Humidity (%)'])
    print('check')
    amb_pressure = float(metadata["Barometric Pressure (Pa)"])

    #region delay, baselines

    metadata['X_O2 Initial'] = X_O2_initial
    metadata['X_CO2 Initial'] = X_CO2_initial
    metadata['X_CO Initial'] = X_CO_initial


    #calculate signal drift using
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

    # Calculate ambient XO2 following ASTM 1354 A.1.4.5
    p_sat_water = 6.1078 * 10**((7.5*amb_temp)/(237.3 + amb_temp)) * 100 #saturation pressure in pa, Magnus approx
    p_h2o = rel_humid/100 * p_sat_water
    X_H2O_initial = p_h2o / amb_pressure

    # Calculate HRR by row

    def get_HRR(row):
        X_O2 = row["O2 (Vol fr)"] 
        X_CO2 = row["CO2 (Vol fr)"]  
        X_CO = row["CO (Vol fr)"]  

        delta_P = row["DPT (Pa)"]
        T_e = row["Te (K)"]

        return calculate_HRR(
            X_O2,
            X_CO2,
            X_CO,
            X_O2_initial,
            X_CO2_initial,
            X_H2O_initial,
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
    
    data["K Smoke (1/m)"] = (1/duct_length) * np.log(1/(data['PD (%)']/100))
    data["HRR (kW)"] = data["HRRPUA (kW/m2)"] * area

    return data, metadata






if __name__ == "__main__":
    parse_dir(INPUT_DIR)