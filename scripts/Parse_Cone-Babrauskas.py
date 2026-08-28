from pathlib import Path
from utils import colorize
import pandas as pd
import json
import shutil
import os
from datetime import datetime
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent         # .../coneDB/scripts
PROJECT_ROOT = SCRIPT_DIR.parent             # .../coneDB 

INPUT_DIR = r"C:\Users\rtg4\Documents\GitHub\NIST-FRG\cone-db\data\preparsed\Box\Babrauskas"
# INPUT_DIR = "\\\\elwood\\733\\internal\\Material-Flam-DB\\Cone\\Box-PreParsed\\Babrauskas"
OUTPUT_DIR_CSV = PROJECT_ROOT / "Exp-Data_Parsed"  / "Box" / "Babrauskas"
OUTPUT_META = PROJECT_ROOT / "Metadata" / "Parsed" / "Box" / "Babrauskas"
LOG_FILE = PROJECT_ROOT / "parse_Babrauskas_log.json"
#LOG2 = PROJECT_ROOT / "parse_Babrauskas_TYPES_log.JSON"

#region parse_dir
# Find/load the pre-parsed CSV files
def parse_dir(input_dir):
    paths = Path(input_dir).glob("**/*.csv")
    paths = list(paths)
    print(paths)
    total_files = len(paths)
    print(colorize(f"Found {total_files} files to parse", "purple"))
    files_parsed = 0
    files_parsed_successfully = 0
    files_skipped = 0
    # track and print parsing success rate
    for path in paths:
        input_meta = path.with_suffix('.json')
        with open(input_meta, "r") as f:
            preparsed_metadata = json.load(f)
        preparsed_date = preparsed_metadata.get("Preparsed", None)
        output_meta = Path(str(input_meta).replace(str(INPUT_DIR), str(OUTPUT_META)))
        output_data = Path(str(path).replace(str(INPUT_DIR), str(OUTPUT_DIR_CSV)))
        if output_meta.exists():
            #IF DOESNT EXIST, CONTINUES. IF DOES, CHECK DATES
            #IF PREPARSED NEWER THAN PARSED, GENERATE CSV AND CLEAR PROCESSING STAGES OF METADATA
            #ADD LOGGING OF THESE ACTIONS TO FRONT OF DATA CORRECTIONS LIST SAYING TO DOUBLE CHECK
            #FOR NOW, KEEP ALL OTHER STUFF FILLED IN
            with open(output_meta, "r") as f:
                metadata = json.load(f)
            parsed_date = metadata.get("Parsed", None)
            if parsed_date > preparsed_date:
            #IF PARSED NEWER THAN PREPARSED   
                if output_data.exists():
                    #Skip if data file also exists
                    print(colorize(f'{path.stem} has already been parsed and is up to date. Skipping Parsing','yellow'))
                    files_skipped += 1
                    continue
                else:
                    #If marked as bad data, do nothing
                    if metadata.get("Bad Data", None):
                        print(colorize(f'{path.stem} has been marked as Bad Data. Skipping Parsing','yellow'))
                        files_skipped += 1
                        continue
                    #Data file missing, just generate thes csv
                    try:
                        files_parsed += 1
                        parse_data(path)      
                    except Exception as e:
                        # log error in md_A_log
                        with open(LOG_FILE, "r", encoding="utf-8") as w:  
                            logfile = json.load(w)
                        logfile.update({
                                str(path.stem) : "Parsing Issue: " + str(e)
                            })
                        with open(LOG_FILE, "w", encoding="utf-8") as f:
                            f.write(json.dumps(logfile, indent=4))

                        print(colorize(f" - Error parsing {path.stem}: {e}\n", "red"))
                        continue
                    files_parsed_successfully += 1
                    print(colorize(f"Parsed {path.stem} successfully\n", "green"))
                    continue
                
            else:
                if metadata['SmURF']:
                    print(colorize(f'{path.stem} has been SmURFed. Please review data corrections and metadata to determine if re-parsing is necessary. Skipping Parsing for now.','yellow'))
                    with open(LOG_FILE, "r", encoding="utf-8") as w:  
                        logfile = json.load(w)
                    logfile.update({
                                str(path.stem) : "SmURFed: Preparsed data is newer than parsed data, but file has been marked as SmURFed. Please review data corrections and metadata to determine if re-parsing is necessary."
                            })
                    with open(LOG_FILE, "w", encoding="utf-8") as f:
                        f.write(json.dumps(logfile, indent=4))
                    files_skipped += 1
                    parse_data(path) #still parse data to generate a csv file if missing, but skip metadata clearing and just log that there is newer data that has not been parsed yet
                    continue

                #IF PREPARSED NEWER THAN PARSED, REGENERATE DATA, CLEAR METADATA PROCESSING STAGES
                print(colorize(f'Data for {path.stem} has been updated since last parse. Re-parsing file.','yellow'))
                #Data file missing, just generate the csv
                try:
                    files_parsed += 1
                    parse_data(path)
                    clear_metadata(input_meta, output_meta, preparsed_date)
                except Exception as e:
                    # log error in md_A_log
                    with open(LOG_FILE, "r", encoding="utf-8") as w:  
                        logfile = json.load(w)
                    logfile.update({
                            str(path.stem) : "Parsing Issue: " + str(e)
                        })
                    with open(LOG_FILE, "w", encoding="utf-8") as f:
                        f.write(json.dumps(logfile, indent=4))

                    print(colorize(f" - Error parsing {path.stem}: {e}\n", "red"))
                    continue
                files_parsed_successfully += 1
                print(colorize(f"Parsed {path.stem} successfully\n", "green"))
                continue
        else:
            #IF NO PARSED METADATA, PARSE AS NEW FILE    
            try:
                files_parsed += 1
                parse_file(path, input_meta, output_meta)
                
            except Exception as e:
                # log error in md_A_log
                with open(LOG_FILE, "r", encoding="utf-8") as w:  
                    logfile = json.load(w)
                logfile.update({
                        str(path.stem) : "Parsing Issue: " + str(e)
                    })
                with open(LOG_FILE, "w", encoding="utf-8") as f:
                    f.write(json.dumps(logfile, indent=4))

                print(colorize(f" - Error parsing {path.stem}: {e}\n", "red"))
                continue
            print(colorize(f"Parsed {path.stem} successfully\n", "green"))
            files_parsed_successfully += 1
    from collections import Counter

    #summary = Counter()
    #with open(LOG2, "r") as logf:
    #    for line in logf:
    #        parts = line.strip().split(',')
    #        if len(parts) >= 3:
    #            route = parts[2]
    #            summary[route] += 1

    # Append the summary to the log file
    #with open(LOG2, "a") as logf:
    #    logf.write("\nSUMMARY OF ROUTE COUNTS:\n")
    #    for route, count in summary.items():
    #        logf.write(f"{route},{count}\n")
    print(colorize(f"Skipped Files:{files_skipped}/{total_files} ({((files_skipped)/total_files) * 100}%)", "blue"))
    if files_parsed > 0:
        print(colorize(f"Files parsed successfully: {files_parsed_successfully}/{files_parsed} ({((files_parsed_successfully)/files_parsed) * 100}%)", "blue"))
    else:
        print(colorize(f"No files needed parsing.", "blue"))
#region parse file   
def parse_file(file_path, input_meta, output_meta):
    parse_data(file_path)
    parse_metadata(input_meta,output_meta)

    
#region parse_plot_data
def parse_data(file_path):
    # extract heat flux from current test
    file_stem = file_path.stem
    meta_file = file_path.with_suffix('.json')
    with open(meta_file, encoding="utf-8") as w:
        metadata = json.load(w)
    mass = metadata["Sample Mass (g)"]
    surf_area = metadata["Surface Area (m2)"]
    df = pd.read_csv(file_path) #Preparsed data columns
    route = None

    #Time/initialize data output dataframe
    data = pd.DataFrame()
    data = df[["Time (s)"]].copy()

    #HRR
    if "HRR (kW)" in df.columns:
        data["HRR (kW)"] = df["HRR (kW)"]
    elif "HRRPUA (kW/m2)" in df.columns: # Hold off on renormalizing until smurfing confirms proper surface area
        data["HRRPUA (kW/m2)"] = df["HRRPUA (kW/m2)"]
        data["HRR (kW)"] = None
    else:
        data["HRR (kW)"] = None

    #Mass
    if "Mass (g)" in df.columns:
        data["Mass (g)"] = df["Mass (g)"]
    elif "Mass LossPUA (kg/m2)" in df.columns: # hold of on renomalizing until smurfing confirms proper surface area and sample mass
        data["Mass LossPUA (kg/m2)"] = df["Mass LossPUA (kg/m2)"]
        data["Mass (g)"] = None
    elif "MLR (g/s)" in df.columns:
        data["MLR (g/s)"] = df["MLR (g/s)"]
        data["Mass (g)"] = None
    elif "MLRPUA (g/s-m2)" in df.columns:
        data["MLRPUA (g/s-m2)"] = df["MLRPUA (g/s-m2)"]
        data["Mass (g)"] = None
    else:
        data["Mass (g)"] = None

    #Flow rate through duct
    if "MFR (kg/s)" in df.columns:
        data["MFR (kg/s)"] = df["MFR (kg/s)"]
    elif "V Duct (m3/s)" in df.columns:
        data["MFR (kg/s)"] = None
        data["V Duct (m3/s)"] = df["V Duct (m3/s)"]
    else:
        data["MFR (kg/s)"] = None

    #Duct T
    if "T Duct (K)" in df.columns:
        # Check if T Duct data is valid (not constant ~273 or similar bad data)
        t_duct_values = df["T Duct (K)"].dropna()
        
        if len(t_duct_values) > 0:
            # Check if all values are within 273 ± 1K (likely bad data)
            mean_value = t_duct_values.mean()
            std_value = t_duct_values.std()
            
            # If mean is around 273 and standard deviation is very small (< 1)
            if abs(mean_value - 273) < 1 and std_value < 1:
                print(colorize(f"Warning: T Duct (K) appears to be bad data (mean={mean_value:.2f}K, std={std_value:.2f}K, constant ~273K), setting to None",'yellow'))
                data["T Duct (K)"] = None
            else:
                data["T Duct (K)"] = df["T Duct (K)"]
        else:
            # All NaN
            data["T Duct (K)"] = None
    else:
        data["T Duct (K)"] = None

    #Gas concentrations
    if "O2 (Vol Fr)" in df.columns:
        data["O2 (Vol fr)"] = df["O2 (Vol Fr)"]
    else:
        data["O2 (Vol fr)"] = None

    if "CO2 (Vol Fr)" in df.columns:
        data["CO2 (Vol fr)"] = df["CO2 (Vol Fr)"]
    elif "CO2 (kg/kg)" in df.columns:
        data["CO2 (Vol fr)"] = None
        data["CO2 (kg/kg)"] = df["CO2 (kg/kg)"]
    else:
        data["CO2 (Vol fr)"] = None

    if "CO (Vol Fr)" in df.columns:
        data["CO (Vol fr)"] = df["CO (Vol Fr)"]
    elif "CO (kg/kg)" in df.columns:
        data["CO (Vol fr)"] = None
        data["CO (kg/kg)"] = df["CO (kg/kg)"]
    else:
        data["CO (Vol fr)"] = None

    if "H2O (Vol Fr)" in df.columns:
        data["H2O (Vol fr)"] = df["H2O (Vol Fr)"]

    if "HCl (Vol Fr)" in df.columns:
        data['HCl (Vol fr)'] = df['HCl (Vol Fr)']

    if "H'Carbs (Vol Fr)" in df.columns:
        data["H'Carbs (Vol fr)"] = df["H'Carbs (Vol Fr)"]
    elif "H'Carbs (kg/kg)" in df.columns:
        data["H'Carbs (kg/kg)"] = df["H'Carbs (kg/kg)"]

    #Smoke
    if 'K Smoke (1/m)' in df.columns:
        data['K Smoke (1/m)'] = df['K Smoke (1/m)']
    elif 'Extinction Area (m2/kg)' in df.columns:
        data['Extinction Area (m2/kg)'] = df['Extinction Area (m2/kg)']
        data['K Smoke (1/m)'] = None
    elif "Smoke Production (m2/s)" in df.columns:
        data["Smoke Production (m2/s)"] = df["Smoke Production (m2/s)"] 
        data['K Smoke (1/m)'] = None

    max_column_order = [
        "Time (s)", "Mass (g)", "HRR (kW)", "MFR (kg/s)", "T Duct (K)", "O2 (Vol fr)", "CO2 (Vol fr)", "CO (Vol fr)",
        "K Smoke (1/m)", "V Duct (m3/s)", "Extinction Area (m2/kg)", "Smoke Production (m2/s)", "Mass Loss (g)", "Mass LossPUA (g/m2)", "MLR (g/s)", "MLRPUA (g/s-m2)",
        "HRRPUA (kW/m2)", "H2O (Vol fr)", "H'Carbs (Vol fr)", "HCl (Vol fr)", "CO2 (kg/kg)", "CO (kg/kg)", "H2O (kg/kg)", "H'carbs (kg/kg)", "HCl (kg/kg)"
                ]

    reordered_data = pd.DataFrame()
    for c in max_column_order:
        if c in data.columns:
            reordered_data[c] = data[c]
    reordered_data.dropna(how='all', inplace=True)               


    #with open(LOG2, "a") as logf:
     #   logf.write(f"{datetime.now().isoformat()},{file_path.name},{route}\n")
    OUTPUT_DIR_CSV.mkdir(parents=True, exist_ok=True)
    data_output_path = OUTPUT_DIR_CSV / str(file_path.name)
    reordered_data.to_csv(data_output_path, index=False)

    print(colorize(f"Generated {data_output_path.name}", "green"))


#region parse_metadata
def parse_metadata(input_meta, output_meta):
    # copy metadata from preparsed to parsed
    Path(output_meta).parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(input_meta, output_meta)
    with open(output_meta, "r") as f:
        metadata = json.load(f)
    #parsed tag
    metadata['Parsed'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    if "Specimen Number" in metadata:
        metadata = {
            ("Replicate" if k == "Specimen Number" else k): (None if k == "Specimen Number" else v)
            for k, v in metadata.items()
        }

    with open(output_meta, "w", encoding="utf-8") as f:
        f.write(json.dumps(metadata, indent=4))
    print(colorize(f"Generated {output_meta.name}", "green"))

def clear_metadata(input_meta, output_meta, preparsed_date):
    #right now input meta is not being used, but if we want to copy new things over keep as an input
    with open(output_meta, "r") as f:
        metadata = json.load(f)
    for stage in ["Preparsed", "Parsed", "Auto Prepared", "Manually Prepared", "SmURF", "Bad Data", "Autoprocessed", "Manually Reviewed Series", "Pass Review", "Published"]:
        metadata[stage] = None

    metadata["Preparsed"] = preparsed_date
    #parsed tag
    metadata['Parsed'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    if len(metadata.get('Data Corrections', [])) == 0:
        metadata['Data Corrections'].append("Metadata processing stages cleared and file re-parsed due to updated preparsed data. Please review other data corrections for accuracy.")
    else:
        metadata['Data Corrections'][0] = "Metadata processing stages cleared and file re-parsed due to updated preparsed data. Please review other data corrections for accuracy."
    with open(output_meta, "w", encoding="utf-8") as f:
        f.write(json.dumps(metadata, indent=4))
    print(colorize(f"Cleared processing stages in {output_meta.name}", "green"))

#region main
if __name__ == "__main__":
    # write new log file at every run
    logfile = {}
    with open(LOG_FILE, "w", encoding="utf-8") as f:
        f.write(json.dumps(logfile, indent=4))
    print("✅ parse_md_A_log.json created.")
    parse_dir(INPUT_DIR)
    