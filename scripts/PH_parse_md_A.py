from pathlib import Path
from utils import colorize
import pandas as pd
import json
import shutil
import os

INPUT_DIR = Path(r"../data/pre-parsed/md_A")
OUTPUT_DIR_CSV = Path(r"../data/parsed/md_A")
PREPARSED_META = Path(r"../metadata/md_A/preparsed")
OUTPUT_META = Path(r"../metadata/md_A/parsed")

LOG_FILE = Path(r"..") / "parse_md_A_log.json"

#region parse_dir
# Find/load the pre-parsed CSV files
def parse_dir(input_dir):
    paths = Path(input_dir).glob("**/*.csv")
    paths = list(paths)
    #paths = list(filter(lambda x: x.stem.endswith("md"), list(paths)))
    print(paths)
    total_files = len(paths)
    print(colorize(f"Found {len(paths)} files to parse", "purple"))
    files_parsed = 0
    files_parsed_successfully = 0

    # track and print parsing success rate
    for path in paths:
        if files_parsed % 20 == 0 and files_parsed != 0:
            print(colorize(f"Files parsed successfully: {files_parsed_successfully}/{files_parsed} ({(files_parsed_successfully/files_parsed) * 100}%)", "blue"))

        try:
            files_parsed += 1
            parse_file(path)
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

    parse_metadata()
    print(colorize(f"Copied all metadata files from {PREPARSED_META} to {OUTPUT_META}\n", "green"))


#region parse file   
def parse_file(file_path):
    parse_data(file_path)
    
#region parse data
def parse_data(file_path):
    df = pd.read_csv(file_path)
    df = df.rename(columns={
        "Q-Dot (kW/m2)" : "HRR (kW/m2)",
        "CO2 (kg/kg)" : "CO2 (Vol fr)",
        "CO (kg/kg)" : "CO (Vol fr)",
        "H2O (kg/kg)" : "H2O (Vol fr)",
        "H'carbs (kg/kg)" : "H'carbs (Vol fr)",
        "M-Dot (g/s-m2)" : "MLR (g/s-m2)",
        "Sum Q (MJ/m2)" : "THR (MJ/m2)"

        })
    
    data = df[
        [
            "Time (s)",
            "HRR (kW/m2)",
            "MLR (g/s-m2)",
            "THR (MJ/m2)"

        ]
    ]
    
    OUTPUT_DIR_CSV.mkdir(parents=True, exist_ok=True)
    data_output_path = OUTPUT_DIR_CSV / str(file_path.name)

    data.to_csv(data_output_path, index=False)

    print(colorize(f"Generated {data_output_path}", "blue"))

#region parse_metadata
def parse_metadata():
    # remove parsed metadata folder if exists
    if os.path.exists(OUTPUT_META):
        shutil.rmtree(OUTPUT_META)
    # copy all metadata files from preparsed metadata folder to parsed metadata folder
    shutil.copytree(PREPARSED_META, OUTPUT_META)


#region main
if __name__ == "__main__":
    # write new log file at every run
    logfile = {}
    with open(LOG_FILE, "w", encoding="utf-8") as f:
        f.write(json.dumps(logfile, indent=4))
    print("✅ parse_md_A_log.json created.")
    parse_dir(INPUT_DIR)