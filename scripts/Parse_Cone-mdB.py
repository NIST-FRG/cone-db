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

INPUT_DIR = PROJECT_ROOT / "data" / "preparsed" / "Box" / "md_B"
OUTPUT_DIR_CSV = PROJECT_ROOT / "Exp-Data_Parsed"  / "Box" / "md_B"
OUTPUT_META = PROJECT_ROOT / "Metadata" / "Parsed" / "Box" / "md_B"
LOG_FILE = PROJECT_ROOT / "parse_md_B_log.json"


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
    files_SmURFed = 0
    bad_files = 0
    # track and print parsing success rate
    for path in paths:
        input_meta = path.with_suffix('.json')
        output_meta = Path(str(input_meta).replace(str(INPUT_DIR), str(OUTPUT_META)))
        if output_meta.exists():
            with open(output_meta, "r") as f:
                metadata = json.load(f)
                if metadata['SmURF'] is not None:
                    files_SmURFed += 1
                    oldname = metadata['Original Testname']
                    newname = metadata ['Testname']
                    print(colorize(f'{oldname} has already been SmURFed to {newname}. Skipping Parsing','blue'))
                    continue
                elif metadata["Bad Data"] is not None:
                    bad_files += 1
                    oldname = metadata['Original Testname']
                    newname = metadata ['Testname']
                    print(colorize(f'{oldname} was deemed bad on {metadata["Bad Data"]}. Skipping Parsing','purple'))
                    continue
        try:
            files_parsed += 1
            parse_file(path, input_meta, output_meta)
            
        except Exception as e:
            # log error in md_A_log
            with open(LOG_FILE, "r", encoding="utf-8") as w:  
                logfile = json.load(w)
            logfile.update({
                    str(path.name): "Parsing Issue: " + str(e)
                })
            with open(LOG_FILE, "w", encoding="utf-8") as f:
	            f.write(json.dumps(logfile, indent=4))

            print(colorize(f" - Error parsing {path}: {e}\n", "red"))
            continue
        print(colorize(f"Parsed {path} successfully\n", "green"))
        files_parsed_successfully += 1

    print(colorize(f"Copied all metadata files from {INPUT_DIR} to {OUTPUT_META}\n", "green"))
    print(colorize(f"Bad Files:{bad_files}/{total_files} ({((bad_files)/total_files) * 100}%)", "blue"))
    print(colorize(f"Files previously SmURFed:{files_SmURFed}/{total_files} ({((files_SmURFed)/total_files) * 100}%)", "blue"))
    print(colorize(f"Files parsed successfully: {files_parsed_successfully}/{files_parsed} ({((files_parsed_successfully)/files_parsed) * 100}%)", "blue"))
    
#region parse file   
def parse_file(file_path, input_meta, output_meta):
    parse_data(file_path)
    parse_metadata(input_meta,output_meta)
'''
#region parse data
def parse_data(file_path):
    # extract heat flux from current test
    file_stem = file_path.stem
    meta_file = str(file_stem) + ".json"
    with open(PREPARSED_META / meta_file, encoding="utf-8") as w:
        metadata = json.load(w)

    df = pd.read_csv(file_path)
    df = df.rename(columns={
        "Q-Dot (kW/m2)" : "HRR (kW/m2)",
        "CO2 (kg/kg)" : "CO2 (Vol %)",
        "CO (kg/kg)" : "CO (Vol %)",
        "H2O (kg/kg)" : "H2O (Vol %)",
        "H'carbs (kg/kg)" : "H'carbs (Vol %)",
        "M-Dot (g/s-m2)" : "MLR (g/s-m2)",
        "Sum Q (MJ/m2)" : "THR (MJ/m2)"

        })

    data = df[
        [
            "Time (s)",
            "HRR (kW/m2)",
            "CO2 (Vol %)",
            "CO (Vol %)",
            "MLR (g/s-m2)",
            "Ex Area (m2/kg)"

        ]
    ]
    
    OUTPUT_DIR_CSV.mkdir(parents=True, exist_ok=True)
    data_output_path = OUTPUT_DIR_CSV / str(file_path.name)

    data.to_csv(data_output_path, index=False)

    print(colorize(f"Generated {data_output_path}", "blue"))

#region parse file   
def parse_file(file_path):
    parse_data(file_path)
'''
    
#region parse_plot_data
def parse_data(file_path):
    # extract heat flux from current test
    file_stem = file_path.stem
    meta_file = file_path.with_suffix('.json')
    with open(meta_file, encoding="utf-8") as w:
        metadata = json.load(w)
    df = pd.read_csv(file_path)

    df["HRRPUA (kW/m2)"] = df["Q-Dot (kW/m2)"]
    df["O2 (Vol fr)"] = None
    df["CO2 (Vol fr)"] = None
    df["CO (Vol fr)"] = None
    df['K Smoke (1/m)'] = None
    df['T Duct (K)'] = None
    df["MLR (kg/s)"] = df["MLR (g/s)"]/1000
    #Derive ksmoke using MLR, V-Duct, and Specific extinction area on fuel pyrolyzate basis (sigma f not sigma s), dont ppull out
    #df["K Smoke (1/m)"] = (df["MLR (kg/s)"]* df["Extinction Area (m2/kg)"])/df["V Duct (m3/s)"]
    df["HRR (kW)"] = None
    data = df[["Time (s)","Mass (g)","HRR (kW)", "MFR (kg/s)","T Duct (K)","O2 (Vol fr)", "CO2 (Vol fr)","CO (Vol fr)",
                "K Smoke (1/m)", "Extinction Area (m2/kg)","HRRPUA (kW/m2)"]].copy()
    
    for gas in ["CO2 (kg/kg)", "CO (kg/kg)", "H2O (kg/kg)", "H'carbs (kg/kg)", "HCl (kg/kg)"]:
        if gas in df.columns:
            data.loc[:, gas] = df[gas]
            
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
    with open(output_meta, "w", encoding="utf-8") as f:
        f.write(json.dumps(metadata, indent=4))




#region main
if __name__ == "__main__":
    # write new log file at every run
    logfile = {}
    with open(LOG_FILE, "w", encoding="utf-8") as f:
        f.write(json.dumps(logfile, indent=4))
    print("✅ parse_md_B_log.json created.")
    parse_dir(INPUT_DIR)
    