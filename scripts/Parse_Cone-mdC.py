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

INPUT_DIR = PROJECT_ROOT / "data" / "preparsed" / "Box" / "md_C"
OUTPUT_DIR_CSV = PROJECT_ROOT / "Exp-Data_Parsed"  / "Box" / "md_C"
OUTPUT_META = PROJECT_ROOT / "Metadata" / "Parsed" / "Box" / "md_C"
LOG_FILE = PROJECT_ROOT / "parse_md_C_log.json"
#LOG2 = PROJECT_ROOT / "parse_md_C_TYPES_log.JSON"


#region parse_dir
# Find/load the pre-parsed CSV files
def parse_dir(input_dir):
    paths = Path(input_dir).glob("**/*.csv")
    paths = list(paths)
    #paths = list(filter(lambda x: x.stem.endswith("md"), list(paths)))
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

    df = pd.read_csv(file_path)
    df["O2 (Vol fr)"] = None
    df["CO2 (Vol fr)"] = None
    df["CO (Vol fr)"] = None
    df["MFR (kg/s)"] = None
    df["K Smoke (1/m)"] = None
    df['T Duct (K)'] = None
    route = None
    if "Mass (kg)" in df.columns:
        #genuinley one test in this tree
        df['MLR (g/s)'] = None
        df["Mass (g)"] = df["Mass (kg)"] * 1000
        df["HRR (kW)"] = None
        if not "CO2 (kg/kg)" in df.columns:
            route = "Mass/noyield"
            print(colorize(f'Warning: {file_stem} does not contain CO2 or CO production data', "yellow"))
            data = df[["Time (s)","Mass (g)","HRR (kW)", "MFR (kg/s)","T Duct (K)","O2 (Vol fr)", "CO2 (Vol fr)","CO (Vol fr)",
                        "K Smoke (1/m)","HRRPUA (kW/m2)"]].copy()
        #NO FILES IN THIS PATTERN BUT KEEP TO BE SAFE
        #else:  
        #    route = "Mass/yield"
        #    data = df[["Time (s)","Mass (g)","HRR (kW)", "MFR (kg/s)", "T Duct (K)", "O2 (Vol fr)", "CO2 (Vol fr)","CO (Vol fr)",
         #                   "K Smoke (1/m)","HRRPUA (kW/m2)","CO2 (kg/kg)", "CO (kg/kg)"]].copy()
    elif "MLR (g/s)" in df.columns:
        #Majority of tests in this tree
        df["Mass (g)"] = None
        df["HRR (kW)"] = None
        print(colorize(f'Warning: {file_stem} only contains mass loss rate data', "yellow"))
        if "CO2 (kg/kg)" in df.columns:
            route = "MLR/yield"
            data =  df[["Time (s)","Mass (g)","HRR (kW)", "MFR (kg/s)","T Duct (K)","O2 (Vol fr)", "CO2 (Vol fr)","CO (Vol fr)",
                            "K Smoke (1/m)","MLR (g/s)", "HRRPUA (kW/m2)", "CO2 (kg/kg)", "CO (kg/kg)"]].copy()
        else:
            route = "MLR/noyield"
            print(colorize(f'Warning: {file_stem} does not contain CO2 or CO production data', "yellow"))
            data =  df[["Time (s)","Mass (g)","HRR (kW)", "MFR (kg/s)","T Duct (K)","O2 (Vol fr)", "CO2 (Vol fr)","CO (Vol fr)",
                "K Smoke (1/m)","MLR (g/s)", "HRRPUA (kW/m2)"]].copy()
    else:
        #Earliest test in this tree
        df["Mass (g)"] = None
        df["HRR (kW)"] = None
        print(colorize(f'Warning: {file_stem} only contains heat relase data, mass loss rate generated from heat of combustion', "yellow"))
        df['MLRPUA (g/s-m2)'] = df['HRRPUA (kW/m2)'] / df['HT Comb (MJ/kg)']
        if not 'CO2 (kg/kg)' in df.columns:
            route = "HRRPUA/noyield"
            print(colorize(f'Warning: {file_stem} does not contain CO2 or CO production data', "yellow"))
            data = df[["Time (s)","Mass (g)","HRR (kW)", "MFR (kg/s)","T Duct (K)","O2 (Vol fr)", "CO2 (Vol fr)","CO (Vol fr)",
                        "K Smoke (1/m)", "MLRPUA (g/s-m2)", "HRRPUA (kW/m2)"]].copy()
        

        #else:
         #   route = "HRRPUA/yield"
          #  df['CO2 ProductionPUA (g/s-m2)'] = df['CO2 (kg/kg)'] * df['MLRPUA (g/s-m2)']
           # df['CO ProductionPUA (g/s-m2)'] = df['CO (kg/kg)'] * df['MLRPUA (g/s-m2)'] 
            #data = df[["Time (s)","Mass (g)","HRR (kW)", "MFR (kg/s)","T Duct (K)","O2 (Vol fr)", "CO2 (Vol fr)","CO (Vol fr)",
             #       "K Smoke (1/m)", "HRRPUA (kW/m2)", "MLRPUA (g/s-m2)", 'CO2 ProductionPUA (g/s-m2)', 'CO ProductionPUA (g/s-m2)']].copy()


   # with open(LOG2, "a") as logf:
   #     logf.write(f"{datetime.now().isoformat()},{file_path.name},{route}\n")

    OUTPUT_DIR_CSV.mkdir(parents=True, exist_ok=True)
    data_output_path = OUTPUT_DIR_CSV / str(file_path.name)

    data.to_csv(data_output_path, index=False)

    print(colorize(f"Generated {data_output_path}", "green"))

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
    print("✅ parse_md_C_log.json created.")
    parse_dir(INPUT_DIR)
    