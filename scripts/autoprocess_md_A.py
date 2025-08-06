from pathlib import Path
from utils import colorize
import pandas as pd
import json
import shutil
import os

INPUT_DIR = Path(r"PH_cone-explorer/data/prepared/md_A")
#OUTPUT_DIR= Path(r"../data/auto-processed/md_A")

def process_dir(input_dir):
    paths = Path(input_dir).glob("**/*.csv")
    paths = list(paths)
    print(paths)
    total_files = len(paths)
    print(colorize(f"Found {len(paths)} files to Autoprocess", "purple"))
    files_parsed = 0
    files_parsed_successfully = 0

    # track and print parsing success rate
    for path in paths:
        if files_parsed % 20 == 0 and files_parsed != 0:
            print(colorize(f"Files parsed successfully: {files_parsed_successfully}/{files_parsed} ({(files_parsed_successfully/files_parsed) * 100}%)", "blue"))

        try:
            files_parsed += 1
            autoprocess(path)
            
        except Exception as e:
            print(colorize(f" - Error parsing {path}: {e}\n", "red"))
            continue
        print(colorize(f"Parsed {path} successfully\n", "green"))
        files_parsed_successfully += 1

def autoprocess(file_path):
    file_stem = file_path.stem
    meta_file = str(file_stem) + ".json"
    meta_path = INPUT_DIR / meta_file
    metadata_json = {}

    #extract existing metadata
    with open(meta_path, "r", encoding="utf-8") as w:  
        metadata_json = json.load(w)

    df = pd.read_csv(file_path)
    avg_mlr = df["MLR (g/s-m2)"].mean(skipna=True)

    metadata_json["MLRPUA"] = avg_mlr

    #update respective test metadata file
    with open(meta_path, "w", encoding="utf-8") as f:
        f.write(json.dumps(metadata_json, indent=4))

#region main
if __name__ == "__main__":
    process_dir(INPUT_DIR)