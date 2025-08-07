import pandas as pd
import re
from pathlib import Path
from io import StringIO
import numpy as np

def read_in_file(file_path):
    print(f"\nProcessing file: {file_path.name}")
    with open(file_path, "r", encoding="utf-8") as file:
        content = file.readlines()
    print(f"Read {len(content)} lines from file")
    return content

# splits file in list of tests, stores as {tests} <key=test_number>
def get_tests(file_contents):
    test_number = -1
    tests = {}
    # Use while loop for lookahead and test detection
    for line in file_contents:
        # normalizes string
        line = str(line).upper().strip()
        # looks for TEST then a digit, re.search returns None if can't find pattern
        test_number_match = re.search("TEST\\s+\\d\\d\\d\\d", line)
        # print(line[test_number_match.start():])
        if (test_number_match is not None) and (line[test_number_match.start():] != test_number):
            # assumes test number will be end of string, gets start of match to EOL
            test_number = line[test_number_match.start():test_number_match.end()]
        if test_number != -1:
            if test_number in tests:
                tests[test_number].append(line)
            else:
                tests[test_number] = [line]
    return tests

def get_metadata(data):
    #use regex
    #wildcard
    # convert metadata segments into list of strings
    
    
    
    
    return data

def get_data(data):
    '''
    data  = list of lines (test)
    '''
    index = 0
    dataStart = -1

    # iterate read in line
    for line in data:
        index += 1
        line = str(line.upper())
        if "PAGE" in line and dataStart == -1:
            dataStart = index
            
    # truncates test to test_data to end [str]
    trimmed_content = data[dataStart:]
    
    # find data_end
    dataEnd = 0
    index = 0
    has_page = False
    for line in trimmed_content:
        if "PAGE" in line:
            has_page = True
        if ("---" == line or index == len(trimmed_content)-1) and has_page:
            has_page = False
            dataEnd = index+1
        index += 1
    trimmed_content = trimmed_content[:dataEnd]
    
    # convert trimmed_content to df
    dataSet = pd.read_csv(StringIO(trimmed_content), sep="|", index_col=-1)
    return dataSet

#     # Detect all header lines (start of each new table section)
#     header_indices = [i for i, line in enumerate(trimmed_lines) if is_header_line(line)]
#     header_indices.append(len(trimmed_lines))  # include end

#     #  Create output folder specific to this file
#     file_output_dir = OUTPUT_DIR / test_name
#     file_output_dir.mkdir(parents=True, exist_ok=True)

#     # Process each section and save as individual CSV
#     for i in range(len(header_indices) - 1):
#         start_idx = header_indices[i]
#         end_idx = header_indices[i + 1]
#         chunk = trimmed_lines[start_idx:end_idx]

#         if not chunk or len(chunk) < 2:
#             continue  # skip short chunks

#         chunk_text = "".join(chunk)
#         try:
#             df = pd.read_csv(StringIO(chunk_text), sep="|", index_col=False)
#             df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
#             df = df.replace(r'\*+', np.nan, regex=True)
#             df = df[~df.apply(lambda row: row.astype(str).str.contains('-').all(), axis=1)]
#             df = rename_columns(df)

#             # Optional: drop second row if it's not numeric
#             if len(df) > 1 and df.iloc[1].apply(lambda x: isinstance(x, str)).all():
#                 df = df.iloc[2:]
#                 print("Removed second row (non-numeric)")

#             # Save to chunk-specific file
#             out_file = file_output_dir / f"{test_name}_chunk{i+1}.csv"
#             df.to_csv(out_file, index=False)
#             print(f"✅ Saved: {out_file}")
#         except Exception as e:
#             print(f"⚠️ Error in chunk {i+1}: {e}")
#  #find other variables in table and add to columns in df then find all occurances of text outside of row one and remove  
    
    


# helper for file with 1 test split across multiple pages 
# def merge_col_on_time(df):
#     df.merge(<col_index>=<time>)
#     return df

def output_json(contents, test_year):
    # Determine output path
    Path(OUTPUT_DIR_JSON / str(test_year)).mkdir(parents=True, exist_ok=True)

    data_output_path = Path(OUTPUT_DIR_JSON) / str(test_year) /f"{Path(file_path).stem}.csv"
    metadata_output_path = Path(OUTPUT_DIR_JSON) / str(test_year) / f"{Path(file_path).stem}.json"

def output_csv(contents, test_year):
    csv = contents.to_csv
    return csv
    
# Define input/output directories
INPUT_DIR = Path("./multi-test.md")
OUTPUT_DIR_MD = Path("../data/test_output/split_md")
OUTPUT_DIR_CSV = Path("../data/test_output/csvs")
OUTPUT_DIR_JSON = Path("../data/test_output/jsons")

# Create output directories if they don't exist
OUTPUT_DIR_MD.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR_CSV.mkdir(parents=True, exist_ok=True)


# read_in_file -> list of every line(str)
# file_contents = read_in_file(INPUT_DIR)
# get_tests -> list of tests
# tests = get_tests(file_contents)
# print(tests.keys())
# print(tests['TEST 2238'])

# Loop through all .md files in INPUT_DIR
for file_path in INPUT_DIR.glob("*.md"):
    
    # read_in_file -> list of every line(str)
    file_contents = read_in_file(file_path)
    
    # get_tests -> list of tests
    tests = get_tests(file_contents)
    for test in tests:
        metadata = get_metadata(test)
        output_json(metadata)
        data = get_data(test)
        output_csv(data)
###########################################



import pandas as pd
import re
from pathlib import Path
from io import StringIO
import json
import sys
from datetime import datetime
from dateutil import parser
import os
import numpy as np
from utils import calculate_HRR, calculate_MFR, colorize
# First argument is the input directory, 2nd argument is the output directory
args = sys.argv[1:]
if len(args) > 2:
    print("""Too many arguments
          Usage: python parse-FTT.py <input_dir> <output_dir>
          Leave empty to use defaults.""")
    sys.exit()

# Relative to the path the script is being run from
# Assumes the script is being run from the root of the repo


## Ex path matching INPUT_DIR: C:Users/user-id/path-to-repo-folder/cone-db/data/raw/FTT
INPUT_DIR = Path("../data/raw/box/A")
# can change later if needed, but wanted to save split md first
## change to autoprocessed folder after done testing
OUTPUT_DIR_MD = Path("../data/test_output/split_md")
OUTPUT_DIR = Path("../data/test_output/csvs")

if len(args) == 2:
    INPUT_DIR = Path(args[0])
    OUTPUT_DIR = Path(args[1])



# Create output directories if they don't exist
OUTPUT_DIR_MD.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Read file content
#with open(INPUT_DIR, "r", encoding="utf-8") as file:
#    lines = file.readlines()

#print(f"Read {len(lines)} lines from file")

#region fix col names
def rename_columns(df):
    new_columns = {}

    for orig_col in df.columns:
        col = orig_col.strip().replace(" ", "").upper()
        if "TIME" in col:
            new_columns[orig_col] = "Time (s)"
        elif "O2" in col:
            new_columns[orig_col] = "O2 (Vol fr)"
        elif "CO2" in col:
            new_columns[orig_col] = "CO2 (Vol fr)"

        elif any(h in col for h in ["HRR", "Q-DOT", "QDOT"]):
            new_columns[orig_col] = "HRR (kW/m2)"
        elif "MFR" in col:
            new_columns[orig_col] = "MFR (kg/s)"
        elif "K_SMOKE" in col:
            new_columns[orig_col] = "k_smoke (1/m)"
        elif "MASSLOSS" in col:
            new_columns[orig_col] = "Mass Loss (g/s-m2)"
        elif any(h in col for h in ["EXTINCTIONAREA", "EXAREA", "EXTAREA"]):
            new_columns[orig_col] = "Extinction Area (m2/kg)"
        elif any(h in col for h in ["HEATOFCOMBUSTION", "HTCOMB"]):
            new_columns[orig_col] = "Heat of Combustion (MJ/kg)"
        elif any(h in col for h in ["HYDROCARBONS", "H'CARBS"]):
            new_columns[orig_col] = "Hydrocarbons (kg/kg)"
        elif "HCL" in col:
            new_columns[orig_col] = "HCl (kg/kg)"
        elif "H2O" in col:
            new_columns[orig_col] = "H2O (kg/kg)"
        elif "CO" in col:
            new_columns[orig_col] = "CO (Vol fr)"
        elif "SUMQ" in col:
            new_columns[orig_col] = "Sum Q (MJ/m2)"

    return df.rename(columns=new_columns)
#region find headers

def is_header_line(line):
    parts = [p.strip() for p in line.split("|") if p.strip()]  # Split line into parts
    
    # Must have a minimum number of columns
    if len(parts) < 3:
        return False

    # If more than half are NOT numeric, maybe it's a header
    num_text = sum(not re.match(r"^\s*[-+]?(\d+(\.\d*)?|\.\d+)\s*$", part) for part in parts)
    if num_text >= len(parts) * 0.6:
        # Bonus: look for common header-like words
        keywords = ['TIME', 'HRR', 'Q', 'CO2', 'CO', 'HCL', 'H2O', 'MASS', 'EXT', 'HEAT', 'CARB']
        return any(kw in part.upper() for part in parts for kw in keywords)

    return False

#region parse_dir
# Find/load the Markdown files
def parse_dir(input_dir):
    paths = Path(input_dir).glob("**/*.md")
    paths = list(paths)
    #paths = list(filter(lambda x: x.stem.endswith("md"), list(paths)))
    print(paths)
    total_files = len(paths)
    print(colorize(f"Found {len(paths)} files to parse", "purple"))
    files_parsed = 0
    files_parsed_successfully = 0
    
    for path in paths:
        if files_parsed % 20 == 0 and files_parsed != 0:
            print(colorize(f"Files parsed successfully: {files_parsed_successfully}/{files_parsed} ({(files_parsed_successfully/files_parsed) * 100}%)", "blue"))
            try:
                files_parsed += 1
                parse_file(path)
            except Exception as e:
                print(colorize(f" - Error parsing {path}: {e}", "red"))
                continue
            print(colorize("Parsed successfully\n", "green"))
            files_parsed_successfully += 1
    
    
    for filename in os.listdir(input_dir):
        if filename.endswith(".md"):
            paths = os.path.join(input_dir, filename)
            if os.path.isfile(paths):
                with open(paths, "r", encoding="utf-8") as file:
                    lines = file.readlines()
                    print(f"Read {len(lines)} lines from file")
                    
#def split_md_df(file_path): maybe move split to parts here
#region parse file   
def parse_file(file_path):

    index = 0
    dataStart = -1
    test_starts = []
    test_sections = {}
    
    print(f"Parsing {file_path}:")
    
    
    with open(file_path, "r", encoding="utf-8") as file:
        lines = file.readlines()
        print(f"Read {len(lines)} lines from file")
       # print(lines)
       
       
    #region split for tests
    # Use while loop for lookahead and test detection
    while index < len(lines):
        line = str(lines[index]).upper().strip()

        if "NBS CONE CALORIMETER" in line:
            test_start = index
            print(f"[\u2713] Found possible test start at line {index + 1}: {line}")

            found_param_sheet = False
            for offset in range(1, 6):
                lookahead_index = index + offset
                if lookahead_index >= len(lines):
                    break
                next_line = lines[lookahead_index].upper().strip()
                cleaned_line = next_line.replace("#", "")
                if "PARAMETER SHEET" in cleaned_line:
                    found_param_sheet = True
                    print(f"Found PARAMETER SHEET at line {lookahead_index + 1}")
                    break

            next_test_start = len(lines)
            for i in range(index + 1, len(lines)):
                if "NBS CONE CALORIMETER" in lines[i].upper():
                    skip = False
                    for offset in range(1, 6):
                        look_idx = i + offset
                        if look_idx >= len(lines):
                            break
                        next_line = lines[look_idx].upper().strip()
                        cleaned = next_line.replace("#", "").strip()
                        if "PARAMETER SHEET" in cleaned:
                            skip = True
                            break
                    if not skip:
                        next_test_start = i
                        break

            print(f"[\u2713] Saving test from line {test_start + 1} to {next_test_start}")
            test_starts.append((test_start, next_test_start))
            index = next_test_start
            continue

        index += 1

    # Save each test section to a markdown file
    for i, (start, end) in enumerate(test_starts):
        test_name = file_path.stem + f"_part{i + 1}"
        test_data = lines[start:end]
        test_sections[test_name] = test_data
 
        #save/split mds
        with open(OUTPUT_DIR_MD / f"{test_name}.md", "w", encoding="utf-8") as test_file:
            test_file.writelines(test_sections[test_name])
            print(f"✅ Test saved to: {OUTPUT_DIR_MD}/{test_name}.md")



                
       #region chunk split
       
    for file_path in OUTPUT_DIR_MD.glob("*.md"):
        print(f"\nProcessing file: {file_path.name}")
    test_name = file_path.stem

    with open(file_path, "r", encoding="utf-8") as file:
        lines = file.readlines()

    # Find start of data section
    dataStart = next((i for i, line in enumerate(lines) if "PAGE" in line.upper()), -1)
    if dataStart == -1:
        print(f"⚠️ No data table start found for {file_path.name}")
        continue

    trimmed_lines = lines[dataStart + 1:]

    # Detect all header lines (start of each new table section)
    header_indices = [i for i, line in enumerate(trimmed_lines) if is_header_line(line)]
    header_indices.append(len(trimmed_lines))  # include end

    #  Create output folder specific to this file
    file_output_dir = OUTPUT_DIR / test_name
    file_output_dir.mkdir(parents=True, exist_ok=True)

    # Process each section and save as individual CSV
    for i in range(len(header_indices) - 1):
        start_idx = header_indices[i]
        end_idx = header_indices[i + 1]
        chunk = trimmed_lines[start_idx:end_idx]

        if not chunk or len(chunk) < 2:
            continue  # skip short chunks

        chunk_text = "".join(chunk)
        try:
            df = pd.read_csv(StringIO(chunk_text), sep="|", index_col=False)
            df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
            df = df.replace(r'\*+', np.nan, regex=True)
            df = df[~df.apply(lambda row: row.astype(str).str.contains('-').all(), axis=1)]
            df = rename_columns(df)

            # Optional: drop second row if it's not numeric
            if len(df) > 1 and df.iloc[1].apply(lambda x: isinstance(x, str)).all():
                df = df.iloc[2:]
                print("Removed second row (non-numeric)")

            # Save to chunk-specific file
            out_file = file_output_dir / f"{test_name}_chunk{i+1}.csv"
            df.to_csv(out_file, index=False)
            print(f"✅ Saved: {out_file}")
        except Exception as e:
            print(f"⚠️ Error in chunk {i+1}: {e}")
 #find other variables in table and add to columns in df then find all occurances of text outside of row one and remove  
    
    
        
def combine_chunks(file_path):
    #for files in folder in OUTPUT_DIR look through chunks to combine columns of same names (time, hrr, etc) make df with all columns given in files (ie row 1 will have all columns named and rows willbe appended in correct order)
    for folder in OUTPUT_DIR.glob("*.csv"):
        for file in folder.glob("*.csv"):
            df = pd.read_csv(file)
            print(df.columns)

    
    #dataSet.shape
#split table where time = 0 again, check if md text has new column headers (should be CO, CO2, O2, Hyd Carb, HCl, H2O or variations in spelling)
def parse_metadata(df):
    ###need to figure out how to do this effectively, search where '=' ?
    #check first section of each part in split_md files
    for file in OUTPUT_DIR_MD.glob("*.md"):
        with open(file, "r", encoding="utf-8") as file:
            lines = file.readlines()
            print(f"Read {len(lines)} lines from file")
            print(lines)

def parse_data(df, metadata):
    ##move below to parse data
    #** dont have mfr yet
    data = df[[
            "Time (s)",
            "Mass (g)",
            "O2 (Vol fr)",
            "CO2 (Vol fr)",
            "CO (Vol fr)",
            "HRR (kW/m2)",
            "MFR (kg/s)",
        ]
    ]

    #save dataset to csv 

if __name__ == "__main__":
    parse_dir(INPUT_DIR)
