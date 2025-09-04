from pathlib import Path
from utils import colorize
import pandas as pd
import json
import shutil
import os
from datetime import datetime
import numpy as np

INPUT_DIR = Path(r"../data/pre-parsed/Box/md_B")
OUTPUT_DIR_CSV = Path(r"../Exp-Data_Parsed/Box/md_B")
PREPARSED_META = Path(r"../Metadata/preparsed/Box/md_B")
OUTPUT_META = Path(r"../Metadata/Parsed/Box/md_B")


LOG_FILE = Path(r"..") / "parse_md_B_log.json"

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
        output_meta = Path(str(path).replace(str(INPUT_DIR), str(OUTPUT_META))).with_suffix('.json')
        input_meta = Path(str(output_meta).replace(str(OUTPUT_META),str(PREPARSED_META)))
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
                    str(path.name)[0:8:1] : "Parsing Issue: " + str(e)
                })
            with open(LOG_FILE, "w", encoding="utf-8") as f:
	            f.write(json.dumps(logfile, indent=4))

            print(colorize(f" - Error parsing {path}: {e}\n", "red"))
            continue
        print(colorize(f"Parsed {path} successfully\n", "green"))
        files_parsed_successfully += 1

    print(colorize(f"Copied all metadata files from {PREPARSED_META} to {OUTPUT_META}\n", "green"))
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
    meta_file = str(file_stem) + ".json"
    with open(PREPARSED_META / meta_file, encoding="utf-8") as w:
        metadata = json.load(w)

    df = pd.read_csv(file_path)
    
    #Check for time discontinuities
    last_time = df["Time (s)"].last_valid_index()
    times =df["Time (s)"].loc[:last_time].values
    start_0 = np.isclose(times[0], 0)
    if not start_0:
        raise Exception(f"Test does not start at 0 seconds, please review preparsed csv file, markdown, and pdf")
    increments = np.diff(times)
    expected_step = np.median(increments)
    #steps continous equal continue changing by the same amount appx (allow for single skip ie times 2) or slight less
    continuous = np.all((increments >= expected_step *.5) & (increments <= expected_step *2))
    if not continuous:
        raise Exception("Test does not have continuous time data, please review preparsed csv file, markdown, and pdf")

    df["HRRPUA (kW/m2)"] = abs(df["Q-Dot (kW/m2)"])
    if "CO2 (kg/kg)" not in df.columns:
        df["CO2 (kg/kg)"] = None
    if "CO (kg/kg)" not in df.columns:
        df["CO (kg/kg)"] = None
    if "H2O (kg/kg)" not in df.columns:
        df["H2O (kg/kg)"] = None
    if "HCl (kg/kg)" not in df.columns:
        df["HCl (kg/kg)"] = None
    if "H'carbs (kg/kg)" not in df.columns:
        df["H'carbs (kg/kg)"] = None
    df["HRR (kW)"] = None
    data = df[["Time (s)","Mass (g)","HRR (kW)", "CO2 (kg/kg)","CO (kg/kg)", "H2O (kg/kg)", "HCl (kg/kg)", "H'carbs (kg/kg)", "HRRPUA (kW/m2)"]]

        
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
    