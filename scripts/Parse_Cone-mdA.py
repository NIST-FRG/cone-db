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

INPUT_DIR = PROJECT_ROOT / "data" / "preparsed" / "Box" / "md_A"
OUTPUT_DIR_CSV = PROJECT_ROOT / "Exp-Data_Parsed"  / "Box" / "md_A"
OUTPUT_META = PROJECT_ROOT / "Metadata" / "Parsed" / "Box" / "md_A"
LOG_FILE = PROJECT_ROOT / "parse_md_A_log.json"
#LOG2 = PROJECT_ROOT / "parse_md_A_TYPES_log.JSON"

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
    df = pd.read_csv(file_path)
    route = None
    data = None
    #initialize unpopulated columns
    df["O2 (Vol fr)"] = None
    df["CO2 (Vol fr)"] = None
    df["CO (Vol fr)"] = None
    df["MFR (kg/s)"] = None
    df['T Duct (K)'] = None
    if "K Smoke (1/m)" not in df.columns:
        df["K Smoke (1/m)"] = None
    else:
        print(colorize(f'K Smoke data found in {file_stem}, retaining original values', "purple"))
    if surf_area == None:
        print(colorize(f'Warning: {file_stem} does not have a defined surface area, data is output per unit area', "yellow"))
        if "Mass Loss (kg/m2)" in df.columns:
            if mass != None:
                route = "massPUA"
                df["MassPUA (g/m2)"] = mass - (df["Mass Loss (kg/m2)"]* 1000) #surface area m2, 1000g/kg
                df["Mass (g)"] = None
                df["HRR (kW)"] = None
                data = df[["Time (s)","Mass (g)","HRR (kW)", "MFR (kg/s)","T Duct (K)","O2 (Vol fr)", "CO2 (Vol fr)","CO (Vol fr)",
                           "K Smoke (1/m)","Extinction Area (m2/kg)", "MassPUA (g/m2)","HRRPUA (kW/m2)"]].copy()
            else: 
                route = "masslossPUA"
                print(colorize(f'Warning: {file_stem} is missing sample mass, mass loss is output', "yellow"))
                df["Mass LossPUA (g/m2)"] = (df["Mass Loss (kg/m2)"]* 1000)
                df["Mass (g)"] = None
                df["HRR (kW)"] = None
                data = df[["Time (s)","Mass (g)","HRR (kW)", "MFR (kg/s)","T Duct (K)","O2 (Vol fr)", "CO2 (Vol fr)","CO (Vol fr)",
                           "K Smoke (1/m)","Extinction Area (m2/kg)","Mass LossPUA (g/m2)","HRRPUA (kW/m2)"]].copy()
        else:
            route = "MLRPUA"
            print(colorize(f'Warning: {file_stem} only contains mass loss rate data', "yellow"))
            df["Mass (g)"] = None
            df["HRR (kW)"] = None
            data = df[["Time (s)","Mass (g)","HRR (kW)", "MFR (kg/s)","T Duct (K)","O2 (Vol fr)", "CO2 (Vol fr)","CO (Vol fr)",
                       "K Smoke (1/m)","Extinction Area (m2/kg)","MLRPUA (g/s-m2)","HRRPUA (kW/m2)"]].copy()
    else:
        df["HRR (kW)"] = surf_area * df["HRRPUA (kW/m2)"]
        if "Mass Loss (kg/m2)" in df.columns:
            if mass != None:
                route = "mass"
                df["Mass (g)"] = mass - (df["Mass Loss (kg/m2)"] * surf_area * 1000) #surface area m2, 1000g/kg
                data = df[["Time (s)","Mass (g)","HRR (kW)", "MFR (kg/s)","T Duct (K)","O2 (Vol fr)", "CO2 (Vol fr)","CO (Vol fr)",
                           "K Smoke (1/m)","Extinction Area (m2/kg)"]].copy()
            else:
                route = "massloss"
                print(colorize(f'Warning: {file_stem} is missing sample mass, mass loss is output', "yellow"))
                df["Mass Loss (g)"] = (df["Mass Loss (kg/m2)"] * surf_area * 1000)
                df["Mass (g)"] = None
                data = df[["Time (s)","Mass (g)","HRR (kW)", "MFR (kg/s)","T Duct (K)","O2 (Vol fr)", "CO2 (Vol fr)","CO (Vol fr)",
                           "K Smoke (1/m)","Extinction Area (m2/kg)","Mass Loss (g)"]].copy()
        else:
            route = "MLR"
            print(colorize(f'Warning: {file_stem} only contains mass loss rate data', "yellow"))
            df["MLR (g/s)"] = df["MLRPUA (g/s-m2)"] * surf_area
            df["Mass (g)"] = None
            data = df[["Time (s)","Mass (g)","HRR (kW)", "MFR (kg/s)","T Duct (K)","O2 (Vol fr)", "CO2 (Vol fr)","CO (Vol fr)",
                       "K Smoke (1/m)","Extinction Area (m2/kg)","MLR (g/s)"]].copy()
            
    for gas in ["CO2 (kg/kg)", "CO (kg/kg)", "H2O (kg/kg)", "H'carbs (kg/kg)", "HCl (kg/kg)"]:
        if gas in df.columns:
            data.loc[:, gas] = df[gas]

    #with open(LOG2, "a") as logf:
     #   logf.write(f"{datetime.now().isoformat()},{file_path.name},{route}\n")
    OUTPUT_DIR_CSV.mkdir(parents=True, exist_ok=True)
    data_output_path = OUTPUT_DIR_CSV / str(file_path.name)

    data.to_csv(data_output_path, index=False)

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
    