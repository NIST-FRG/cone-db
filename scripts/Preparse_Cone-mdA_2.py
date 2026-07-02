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
import traceback
import shutil
from utils import calculate_HRR, calculate_MFR, colorize



#HANDLING NEW MD_B FORMAT: FOR NEW LLAMMA'D FILES
#Path Handling: Relative to this script's location
SCRIPT_DIR = Path(__file__).resolve().parent         # .../coneDB/scripts
PROJECT_ROOT = SCRIPT_DIR.parent             # .../coneDB 

INPUT_DIR = PROJECT_ROOT / "data" / "raw" / "Box" / "md_A_new"### WILL BE FIREDATA IN BOX SUBFOLDER, (firedata/flammabilitydata/cone/Box/md_A_new)
OUTPUT_DIR = PROJECT_ROOT / "data" / "preparsed" / "Box" / "md_A"
LOG_FILE = PROJECT_ROOT / "preparse_md_A_2_log.json"



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
    files_parsed_fully = 0
    files_parsed_partial = 0
    
    for path in paths:
        files_parsed += 1
        try:
            pct = parse_file(path)
        except Exception as e:
            with open(LOG_FILE, "r", encoding="utf-8") as w:  
                logfile = json.load(w)
            logfile.update({
                    str(path) : str(e)
                })
            with open(LOG_FILE, "w", encoding="utf-8") as f:
	            f.write(json.dumps(logfile, indent=4))

            print(colorize(f" - Error parsing {path}: {e}\n", "red"))
            #out_path = Path(str(path).replace('md_B', 'md_B_bad'))
            #out_path.parent.mkdir(parents=True, exist_ok=True)
            #shutil.move(path, out_path)
            continue
        out_path = False
        if pct == 100:
            print(colorize(f"Parsed {path} successfully\n", "green"))
            files_parsed_fully += 1
        elif pct == 0 or pct == None:
            print(colorize(f"{path} could not be parsed", "red"))
            #out_path = Path(str(path).replace('md_B', 'md_B_bad'))
        else:
            print(colorize(f'{pct}% of tests in {path} parsed succesfully\n', 'yellow'))
            files_parsed_partial += 1
            #out_path = Path(str(path).replace('md_B', 'md_B_partial'))

        # If output path is set, ensure the directory exists and copy
        #if out_path:
         #   out_path.parent.mkdir(parents=True, exist_ok=True)
          #  shutil.move(path, out_path)
    print(colorize(f"Files pre-parsed fully: {files_parsed_fully}/{files_parsed} ({((files_parsed_fully)/files_parsed) * 100}%)", "blue"))
    print(colorize(f"Files pre-parsed partially: {files_parsed_partial}/{files_parsed} ({((files_parsed_partial)/files_parsed) * 100}%)", "blue"))
    
    '''
    # printing number of lines read
    for filename in os.listdir(input_dir):
        if filename.endswith(".md"):
            paths = os.path.join(input_dir, filename)
            if os.path.isfile(paths):
                with open(paths, "r", encoding="utf-8") as file:
                    lines = file.readlines()
                    print(f"Read {len(lines)} lines from file")
    '''

#def split_md_df(file_path): maybe move split to parts here
#region parse file   
def parse_file(file_path):
    index = 0
    dataStart = -1
    test_starts = []
    test_sections = {}
    
    print(colorize(f"Parsing {file_path.name}:", "yellow"))
    
    # read lines in file
    with open(file_path, "r", encoding="utf-8") as file:
        lines = file.readlines()
        print(f"Read {len(lines)} lines from file")
       # print(lines)

    # separating tests within the file
    try: 
        tests = get_tests(lines)
        numtests = len(tests)
        parsed = 0
    except Exception as e:
        with open(LOG_FILE, "r", encoding="utf-8") as w:  
            logfile = json.load(w)
        logfile.update({
              f"{file_path.name}": f"{e}"
        })
        with open(LOG_FILE, "w", encoding="utf-8") as f:
	        f.write(json.dumps(logfile, indent=4))
        print(colorize(f" - Error parsing {file_path.name}: {e}\n", "red"))
        return
    for test in tests:
        try:
            # for each test, separate data from metadata
            test_data_df, metadata = get_data(tests[test])
            # generate test data csv
            data_df,test_filename = parse_data(test_data_df,test,file_path.name)
            data_df = data_df.replace([np.inf, -np.inf], np.nan).dropna(how='all')
            test_name = f"{test_filename}.csv"
            output_path = OUTPUT_DIR / test_name
            if output_path.exists():
                old_df = pd.read_csv(output_path)
                # Compare old and new dataframes
                if old_df.equals(data_df):
                    print(colorize(f"{test_filename} already exists and is identical. Skipping generation.", "blue"))
                    parsed += 1
                    continue
                else:
                    print(colorize(f"{test_filename} already exists but differs. Overwriting with new data.", "yellow"))
            # parse through and generate metadata json file
            status = parse_metadata(metadata,test_filename)
            if status == None:
                data_df.to_csv(output_path, index=False)
                print(colorize(f"Generated {output_path}", "blue"))
                parsed += 1
            elif status == "SmURF" or status == "Bad":
                parsed +=1
        except Exception as e:
            tb_list = traceback.extract_tb(e.__traceback__)
            fail = None
            for tb in reversed(tb_list):
                if "Preparse_Cone-mdB" in tb.filename and "get_number" not in tb.name:
                    fail = tb
                    break
            if not fail:
                print(tb_list)
                fail = tb_list[0] 
            location = f"{fail.filename.split("\\")[-1]}:{fail.lineno} ({fail.name})"
            # log error in md_A_log
            with open(LOG_FILE, "r", encoding="utf-8") as w:  
                logfile = json.load(w)
            logfile.update({
                 f"{file_path.name}-{test}": f"{e} @ {location}"
            })
            with open(LOG_FILE, "w", encoding="utf-8") as f:
	            f.write(json.dumps(logfile, indent=4))

            print(colorize(f" - Error parsing {test}: {e}\n", "red"))
            continue
    pct_parsed = (parsed / numtests) *100
    return pct_parsed
   

####### separate tests in file #######
#region get_tests - MODIFIED FOR NEW FORMAT
def get_tests(file_contents):
    """
    Splits file into list of tests by identifying test headers with format:
    MATERIAL NAME (TYPE) HeatFlux KW/M2 ORIENTATION. (TestNumber)
    """
    test_number = -1
    tests = {}
    
    for i in range(len(file_contents)):
        line = str(file_contents[i]).upper().strip()
        
        test_match = re.search(r"\((\d{3,4}[a-zA-Z]?)\)\s*$", line)
        if test_match is None:
            # Add parentheses to create capture group
            test_match = re.search(r"TEST\s+(\d{4})", line)
        
        if test_match is not None:
            raw = test_match.group(1)
            test_number = f"Test {raw.zfill(4)}"
            
            # Initialize new test
            if test_number not in tests:
                tests[test_number] = []
        
        # Add line to current test if one has been identified
        if test_number != -1:
            tests[test_number].append(line)
    
    print(colorize(f"Found {len(tests)} tests: {list(tests.keys())}", "cyan"))
    return tests
    

####### separate metadata from test data #######
# outputting dataframe to csv file
#region get_data - MODIFIED FOR NEW FORMAT
def get_data(data):
    """
    Separates metadata from tabular data.
    Groups tables by column set, concatenates within groups, merges between groups by TIME.
    Merges duplicate columns by taking the first non-NaN value.
    Summary tables go into metadata.
    """
    dataStart = -1
    metadata = []
    table_blocks = []
    current_table = []
    in_summary_table = False
    
    for i, line in enumerate(data):
        line_upper = str(line).upper().strip()
        
        # Check if this is a table header (has TIME and pipe separator)
        if "TIME" in line_upper and "|" in line:
            # Check if this is a summary table
            if any(keyword in line_upper for keyword in [
                "MAX", "PARAMETER", "YIELD", "PEAK", "AVERAGE", 
                "IGNITION", "TOTAL HEAT RELEASE", "MASS WEIGHTED"
            ]):
                # This is a summary table - add to metadata
                in_summary_table = True
                metadata.append(line)
                continue
            else:
                # This is a time-series data table
                in_summary_table = False
                if dataStart == -1:
                    dataStart = i
                if current_table:
                    table_blocks.append(current_table)
                    current_table = []
                current_table.append(line)
        elif in_summary_table:
            # Continue capturing summary table lines as metadata
            # Stop when we hit a non-table line (no pipe or empty)
            if "|" in line or line.strip().startswith("*"):
                metadata.append(line)
            else:
                in_summary_table = False
                metadata.append(line)
        elif dataStart != -1 and not in_summary_table:
            # We're in a data table
            current_table.append(line)
        else:
            # Regular metadata
            metadata.append(line)
    
    if current_table:
        table_blocks.append(current_table)
    
    print(colorize(f"Found {len(table_blocks)} time-series table blocks", "cyan"))
    print(colorize(f"Captured {len(metadata)} metadata lines", "cyan"))
    
    # Process each table block separately
    dfs = []
    for idx, table_block in enumerate(table_blocks):
        print(colorize(f"Processing table block {idx + 1}...", "yellow"))
        
        filtered_table = []
        for line in table_block:
            # Skip empty lines and separator lines
            if line.strip().replace('-', '').replace('|', '').replace(' ', '') == '':
                continue
            if not line.strip():
                continue
            filtered_table.append(line)
        
        # Skip if table is too small (likely not real data)
        if len(filtered_table) < 3:
            print(colorize(f"Skipping table block {idx + 1}: too few rows", "yellow"))
            continue
        
        try:
            pd_format_test_data = StringIO("\n".join(filtered_table))
            block_df = pd.read_csv(pd_format_test_data, sep="|")
            block_df = block_df.iloc[:, 1:-1]
            
            # Skip tables with empty or whitespace-only column names
            if any(col.strip() == '' for col in block_df.columns):
                print(colorize(f"Skipping table block {idx + 1}: contains empty columns (moving to metadata)", "yellow"))
                # Add this table to metadata instead
                metadata.extend(filtered_table)
                continue
            
            # Normalize column names with tracking for duplicates
            new_columns = []
            
            for col in block_df.columns:
                col_upper = col.upper().strip()
                
                # Skip empty columns
                if not col_upper:
                    print(colorize(f"Skipping empty column in table {idx + 1}", "yellow"))
                    continue
                
                new_name = None
                
                if "TIME" in col_upper:
                    new_name = "Time (s)"
                elif "Q-DOT" in col_upper or "Q DOT" in col_upper:
                    new_name = "HRRPUA (kW/m2)"
                elif "SUM Q" in col_upper or "SUMQ" in col_upper:
                    new_name = "THRPUA (MJ/m2)"
                elif "M-DOT" in col_upper or "M DOT" in col_upper:
                    new_name = "MLRPUA (g/s-m2)"
                elif "MASS" in col_upper and "LOSS" in col_upper:
                    if "MLRPUA (g/s-m2)" not in new_columns:
                        new_name = "MLRPUA (g/s-m2)"
                    else:
                        new_name = "Mass Loss (kg/m2)"
                elif "H" in col_upper and ("COM" in col_upper or "CON" in col_upper):
                    new_name = "HT Comb (MJ/kg)"
                # CHECK CO2 FIRST (before CO)
                elif "CO2" in col_upper or "CO₂" in col_upper or "C02" in col_upper:
                    new_name = "CO2 (kg/kg)"
                # CHECK H2O BEFORE OTHER H2 patterns
                elif "H2O" in col_upper or "H₂O" in col_upper or "H20" in col_upper:
                    new_name = "H2O (kg/kg)"
                # NOW CHECK CO (after CO2 is ruled out)
                elif ("CO" in col_upper or "C0" in col_upper) and not any(x in col_upper for x in ["2", "H", "S", "M", "CARB", "CARS", "CARD"]):
                    new_name = "CO (kg/kg)"
                elif "CARB" in col_upper or "CARS" in col_upper or "CARD" in col_upper:
                    new_name = "H'carbs (kg/kg)"
                elif "HCL" in col_upper:
                    new_name = "HCl (kg/kg)"
                elif "EX" in col_upper and "AREA" in col_upper:
                    new_name = "Extinction Area (m2/g)"
                elif "EPSILON" in col_upper:
                    if "Epsilon One (kg/kg)" not in new_columns:
                        new_name = "Epsilon One (kg/kg)"
                    else:
                        new_name = "Epsilon Two (kg/kg)"
                elif "TEOM" in col_upper: 
                    new_name = "TEOM (g/s)"
                elif "KS" in str(col.replace(" ", "")):
                    new_name = "Mass Smoke Extinction Area (m2/g)"
                elif "EXTCOEFF" in col_upper:
                    new_name = "K Smoke (1/m)"
                else:
                    # Throw error for illegal columns
                    msg = f'Illegal Column Detected in table {idx + 1}: "{col}" (normalized: "{col_upper}")'
                    raise Exception(msg)
                
                # Track duplicates with temporary suffix for merging later
                if new_name and new_name in new_columns:
                    counter = 2
                    temp_name = f"{new_name}__dup{counter}"
                    while temp_name in new_columns:
                        counter += 1
                        temp_name = f"{new_name}__dup{counter}"
                    new_columns.append(temp_name)
                    print(colorize(f"Warning: Duplicate column '{new_name}' found as '{col}' in table {idx + 1}, will merge", "yellow"))
                elif new_name:
                    new_columns.append(new_name)
            
            # If no valid columns found, skip
            if not new_columns:
                print(colorize(f"Table {idx + 1} has no valid columns. Moving to metadata.", "yellow"))
                metadata.extend(filtered_table)
                continue
            
            block_df.columns = new_columns
            
            # **CONVERT TIME COLUMN TO NUMERIC FIRST**
            if "Time (s)" in block_df.columns:
                block_df["Time (s)"] = pd.to_numeric(block_df["Time (s)"], errors='coerce')
            
            # Merge duplicate columns within this table
            block_df = merge_duplicate_columns(block_df)
            
            # **ENSURE TIME IS NUMERIC AFTER MERGE TOO**
            if "Time (s)" in block_df.columns:
                block_df["Time (s)"] = pd.to_numeric(block_df["Time (s)"], errors='coerce')
            
            dfs.append(block_df)
            print(colorize(f"Table {idx + 1}: {block_df.shape[0]} rows, {block_df.shape[1]} columns", "blue"))
            print(colorize(f"Columns: {list(block_df.columns)}", "cyan"))
            
        except Exception as e:
            # Re-raise the exception to stop processing
            print(colorize(f"Error parsing table block {idx + 1}: {e}", "red"))
            raise
    
    if not dfs:
        raise Exception("No valid time-series tables found in data")
    
    # **VERIFY ALL TIME COLUMNS ARE NUMERIC BEFORE GROUPING**
    for idx, df in enumerate(dfs):
        if "Time (s)" in df.columns:
            if df["Time (s)"].dtype != 'float64':
                print(colorize(f"Warning: Converting Time column in table {idx + 1} from {df['Time (s)'].dtype} to float64", "yellow"))
                dfs[idx]["Time (s)"] = pd.to_numeric(dfs[idx]["Time (s)"], errors='coerce')
    
    # Group tables by column set
    column_groups = {}
    for idx, df in enumerate(dfs):
        col_key = tuple(sorted(df.columns))
        if col_key not in column_groups:
            column_groups[col_key] = []
        column_groups[col_key].append((idx, df))
    
    print(colorize(f"Grouped into {len(column_groups)} column groups", "cyan"))
    
    # Concatenate within each group
    merged_groups = []
    for group_idx, (col_key, group) in enumerate(column_groups.items()):
        print(colorize(f"Merging column group {group_idx + 1} ({len(group)} tables)...", "yellow"))
        
        merged_group = group[0][1].copy()
        
        for table_idx in range(1, len(group)):
            table = group[table_idx][1].copy()
            table = table.iloc[1:].reset_index(drop=True)
            merged_group = pd.concat([merged_group, table], axis=0, ignore_index=True)
        
        merged_groups.append(merged_group)
        print(colorize(f"Group {group_idx + 1}: {merged_group.shape[0]} rows, {merged_group.shape[1]} columns", "green"))
    
    # Merge between groups horizontally by TIME
    print(colorize(f"Merging {len(merged_groups)} groups horizontally by TIME...", "yellow"))
    
    test_data_df = merged_groups[0].copy()
    
    for group_idx in range(1, len(merged_groups)):
        group_df = merged_groups[group_idx].copy()
        
        # **ENSURE BOTH TIME COLUMNS ARE NUMERIC BEFORE MERGE**
        if "Time (s)" in test_data_df.columns:
            test_data_df["Time (s)"] = pd.to_numeric(test_data_df["Time (s)"], errors='coerce')
        if "Time (s)" in group_df.columns:
            group_df["Time (s)"] = pd.to_numeric(group_df["Time (s)"], errors='coerce')
        
        # Get non-TIME columns
        group_data_cols = [c for c in group_df.columns if c != "Time (s)"]
        
        # Check for conflicts
        conflicting_cols = set(test_data_df.columns) & set(group_data_cols)
        if conflicting_cols:
            print(colorize(f"Warning: Merging duplicate columns across groups: {conflicting_cols}", "yellow"))
        
        # Merge on TIME
        test_data_df = pd.merge(
            test_data_df,
            group_df[["Time (s)"] + group_data_cols],
            on="Time (s)",
            how="outer",
            suffixes=('', '__crossgroup')
        )
        
        print(colorize(f"After merging group {group_idx + 1}: {test_data_df.shape[0]} rows, {test_data_df.shape[1]} columns", "green"))
    
    # Final merge of any cross-group duplicates
    test_data_df = merge_duplicate_columns(test_data_df)
    
    print(colorize(f"Final merged table: {test_data_df.shape[0]} rows, {test_data_df.shape[1]} columns", "green"))
    print(colorize(f"Final columns: {list(test_data_df.columns)}", "green"))
    
    return test_data_df, metadata


def merge_duplicate_columns(df):
    """
    Merges columns with the same base name (ignoring __dup or __crossgroup suffixes).
    Takes first non-NaN value across duplicate columns for each row.
    """
    # Find all unique base column names
    base_names = {}
    for col in df.columns:
        base = col.split('__')[0]  # Remove suffixes like __dup2 or __crossgroup
        if base not in base_names:
            base_names[base] = []
        base_names[base].append(col)
    
    # Merge duplicates
    merged_df = pd.DataFrame()
    for base, cols in base_names.items():
        if len(cols) == 1:
            # No duplicates, keep as is
            merged_df[base] = df[cols[0]]
        else:
            # Multiple columns with same base name - merge them
            print(colorize(f"  Merging {len(cols)} columns into '{base}'", "cyan"))
            # Combine by taking first non-NaN value across the row
            merged_df[base] = df[cols].bfill(axis=1).iloc[:, 0]
    
    return merged_df


def parse_data(data_df, test, file_name):
    """
    Minimal cleaning: remove header rows, convert to numeric, export.
    """
    
    # Remove any header rows that got parsed as data
    def is_header_row(row):
        """Check if a row contains only header text/units"""
        header_keywords = ['TIME', 'S', 'KW/M2', 'MJ/M2', 'MJ/KG', 'GRAMS', 'G/S', 
                          'KG/KG', 'KG/S', 'M3/S', 'M2/KG', 'DOT', 'DUCT', 'AREA', 'SOOT', 'CARB', 'HCL']
        
        cell_count = 0
        header_count = 0
        for cell in row:
            cell_str = str(cell).upper().strip()
            if cell_str and cell_str != 'nan':
                cell_count += 1
                if any(kw in cell_str for kw in header_keywords):
                    header_count += 1
        
        return cell_count > 0 and (header_count / cell_count > 0.5)
    
    data_df = data_df[~data_df.apply(is_header_row, axis=1)]
    data_df = data_df.reset_index(drop=True)
    
    # Clean cell values: remove asterisks and dashes, convert to numeric
    def clean_cells(col):
        col_stripped = col.astype(str).str.strip()
        col_cleaned = col_stripped.apply(lambda x: np.nan if ("*" in str(x) or str(x).strip() == '-') else x)
        return pd.to_numeric(col_cleaned, errors='coerce')
    
    data_df = data_df.apply(clean_cells)
    last_time = data_df['Time (s)'].last_valid_index()
    times =data_df['Time (s)'].loc[:last_time].values
    start_0 = np.isclose(times[0], 0)
    if not start_0:
        raise Exception(f"Test does not start at 0 seconds, please review markdown and pdf")
    increments = np.diff(times)
    expected_step = np.median(increments)
    #steps continous equal continue changing by the same amount appx (allow for single skip ie times 2) or slight less
    continuous = np.all((increments >= expected_step *.1) & (increments <= expected_step *5))
    if not continuous:
        raise Exception("Test does not have continuous time data, please review markdown and pdf")
    for c in data_df.columns:
        data = data_df[c].loc[:data_df[c].last_valid_index()].values
        if len(data) > len(times) and not np.all(np.isnan(data)):
            raise Exception(f"Column {c} exceeds the length of time in the test, please review markdown and pdf")
    # Generate test filename
    test_name = test.casefold().replace(" ", "")
    test_filename = test_name + "_" + file_name[0:8]
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    print(colorize(f"Final columns: {list(data_df.columns)}", "green"))
    
    return data_df, test_filename
####### metadata clean and output functions #######
#region parse_metadata
# clean and output metadata as json
def parse_metadata(input,test_name):
    meta_filename = test_name + ".json"
    meta_path = OUTPUT_DIR / meta_filename
    metadata_json = {}
    metadata = []

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # checking for existing test metadata file 
    if meta_path.exists():
        with open(meta_path, "r", encoding="utf-8") as w:  
            metadata_json = json.load(w)
        if metadata_json['SmURF'] is not None:
                    files_SmURFed += 1
                    oldname = metadata_json['Original Testname']
                    newname = metadata_json['Testname']
                    print(colorize(f'{oldname} has already been SmURFed to {newname} on {metadata_json["SmURF"]}. Skipping Preparsing','blue'))
                    return 'Smurf'
        elif metadata_json["Bad Data"] is not None:
                    bad_files += 1
                    oldname = metadata_json['Original Testname']
                    print(colorize(f'{oldname} was deemed bad on {metadata_json["Bad Data"]}. Skipping Preparsing','purple'))
                    return 'Bad'
    for line in input:
        # Preprocess line to remove excessive whitespace after '='
        line = re.sub(r'=\s*', '= ', line)  # Replace '=' followed by whitespace with '= '
        line = re.sub(r'\s*=', '=', line) # Remove whitespace before "="
        # finds all space blocks separating potential metadata values
        # assumes metadata blocks are separated by at least 3 whitespaces
        match_whitespace = re.search("\\s{3,}", line)
        match_semicolon = re.search(";", line)
        while match_whitespace is not None or match_semicolon is not None:

            # checks which match is closer to the beginning (grabs only 1 metadata block)
            if match_whitespace is not None:
                match = match_whitespace
            else:
                match = match_semicolon
            if match_whitespace is not None and match_semicolon is not None:
                if match_semicolon.start() < match_whitespace.start():
                    match = match_semicolon
                else: 
                    match = match_semicolon

            metadata.append(line[:match.start()])
            # trims line up to end of last match
            line = line[match.end():]
            match_whitespace = re.search("\\s{3,}", line)
            match_semicolon = re.search(";", line)
        if line.strip() != "":
            metadata.append(line)

    # metadata = list of metadata blocks as str
    #print(metadata)

    
    ############ finding metadata fields ############
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
    'Preparsed',
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
    "Ignition Source",
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
    'Peak HRRPUA (kW/m2)', 'Peak HRRPUA Outlier',
    'Total Heat Release (MJ/m2)', 'Total Heat Release Outlier',
    'Effective Heat Of Combustion (kJ/g)', 'Effective Heat Of Combustion Outlier',
    'Average Specific Extinction Area (m2/kg)', 'Average Specific Extinction Area Outlier',
    'Smoke Production Pre-ignition (m2/m2)','Smoke Production Pre-ignition Outlier',
    'Smoke Production Post-ignition (m2/m2)','Smoke Production Post-ignition Outlier',
    'Smoke Production Total (m2/m2)','Smoke Production Total Outlier',
    'Y_Soot (g/g)', 'Y_Soot Outlier',
    'Y_CO2 (g/g)', 'Y_CO2 Outlier',
    'Y_CO (g/g)', 'Y_CO Outlier',
    'Fire Growth Potential (m2/J)', 'Fire Growth Potential Outlier',
    'Ignition Energy (MJ/m2)', 'Ignition Energy Outlier',
    "t_flameout (s)","t_flameout Outlier",
    'Comments', 'Data Corrections'
        ]
    for key in expected_keys:
        metadata_json.setdefault(key, None)

    metadata_json["Comments"] = []
    prev_item = None
    for item in metadata:
        if metadata.index(item) == 0:
            metadata_json["Institution"] = item
        elif "IRRADIANCE" in item:
            metadata_json["Heat Flux (kW/m2)"] = get_number(item,"int")
        elif "TEST" in prev_item and not metadata_json.get("Material Name"):
            metadata_json["Material Name"] = item    
        elif "HOR" in item:
            metadata_json["Orientation"] = "HORIZONTAL"
        elif "VERT" in item:
            metadata_json["Orientation"] = "VERTICAL"
        elif "CALIBRATION" in item:
            metadata_json["C Factor"] = get_number(item[3:],"flt")
        elif "SPARK IGN" in item and "HOLDER" in item and ("Mask" in item or "Grid" in item):
            metadata_json["Ignition Source"] = "Spark Igniter"
            metadata_json['Edge Frame'] = True
            metadata_json["Grid"] = True
        elif "SPARK IGN" in item and ("HOLDER" in item or "FRAME" in item):
            metadata_json["Ignition Source"] = "Spark Igniter"  
            metadata_json['Edge Frame'] = True
        elif "NO SPARK" in item:
            metadata_json["Ignition Source"] = "No Source"  
        elif "NO GRID" in item or "NO MASK" in item:
            metadata_json["Grid"] = False
        elif "SPARK IGN" in item or "SPARKER" in item:
            metadata_json["Ignition Source"] = "Spark Igniter"
        elif "GRID" in item or "MASK" in item:
            metadata_json["Grid"] = True
        elif "W/OPILOT" in item.replace(" ", ""):
            metadata_json["Ignition Source"] = "No Source"
        elif "PILOT" in item:
            metadata_json["Ignition Source"] = "Pilot Flame"
        elif "INITIAL MASS" in item and "FRACTION" not in item:
            metadata_json["Sample Mass (g)"] = get_number(item[3:],"flt")
        elif "FINAL MASS" in item and "FRACTION" not in item:
            metadata_json["Residual Mass (g)"] = get_number(item[3:],"flt")
        elif "AREA OF SAMPLE" in item and not metadata_json.get("Surface Area (m2)"):
            metadata_json["Surface Area (m2)"] = get_number(item,"flt")
            if metadata_json['Surface Area (m2)'] == 0.01 and metadata_json["Edge Frame"] is None:
                metadata_json['Edge Frame'] = False
        elif "SOOT AVERAGE" in item:
            metadata_json["Y_Soot (g/g)"] = get_number(item,"flt")
        elif item.find("CONVERSION FACTOR") == 0:
            hoc_o2_kJkg = get_number(item,"int")
            hoc = hoc_o2_kJkg / 1000
            metadata_json["Heat of Combustion O2 (MJ/kg)"] = hoc
        elif "TIME TO IGNITION" in item:
            metadata_json["t_ignition (s)"] = get_number(item,"int")
        elif "TEST" in item:
            match = re.search(r'TEST\s+(\d{4})', item)
            if match:
                metadata_json["Specimen Number"] = int(match.group(1))
        elif "INITIAL WEIGHT" in item and not metadata_json.get("Sample Mass (g)"):
            metadata_json["Sample Mass (g)"] = get_number(item[3:],"flt")
        elif re.search(r'\d+\s+[A-Z]{3}\s+\d{4}', item) is not None:
            metadata_json["Test Date"] = item
        
        metadata_json["Comments"].append(item)
        prev_item = item

    metadata_json['Original Testname'] = test_name
    metadata_json['Instrument'] = "NBS Cone Calorimeter"
    metadata_json['Preparsed'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    metadata_json["Original Source"] = "Box/md_A"
    metadata_json['Data Corrections'] =[]
    if metadata_json['Surface Area (m2)'] == 0.01 and metadata_json["Edge Frame"] is None:
        metadata_json['Edge Frame'] = False
    elif metadata_json["Edge Frame"] is None and metadata_json['Surface Area (m2)'] is not None:
        if metadata_json['Surface Area (m2)'] <= 0.009 and metadata_json['Surface Area (m2)'] > 0.008:
            metadata_json['Edge Frame'] = True
    #update respective test metadata file
    with open(meta_path, "w", encoding="utf-8") as f:
        f.write(json.dumps(metadata_json, indent=4)) 
    print(colorize(f"Generated {meta_path}", "blue"))
    return None
    

    # metadata table checker
    # if line[0] == | then until not | at line[0] add to a table


    # # Determine output path
    # Path(OUTPUT_DIR_JSON / str(test_year)).mkdir(parents=True, exist_ok=True)

    # data_output_path = Path(OUTPUT_DIR_JSON) / str(test_year) /f"{Path(file_path).stem}.csv"
    # metadata_output_path = Path(OUTPUT_DIR_JSON) / str(test_year) / f"{Path(file_path).stem}.json"

#region helpers
#get number(int,float,exponent,)
def get_number(item, num_type):
    number = None
    match num_type:
        case "int":
            match = re.search(r'\d+', item)
            if match:
                number = int(match.group())
        case "flt":
            match = re.search(r'\d*\.\d+', item)
            if match:
                number = float(match.group())
        case "exp":
            match = re.search(r'\d*\.\d+[E][+-]*\d', item)
            if match:
                number = str(match.group())

    return number

#get content after the "="            
def get_field(item):
    index = item.find('=')
    field = item[index+1:]
    field = field.strip()
    return field       

#region main
if __name__ == "__main__":
    # write new log file at every run
    LOG_DIR = Path(r"../logs/")
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    logfile = {}
    with open(LOG_FILE, "w", encoding="utf-8") as f:
        f.write(json.dumps(logfile, indent=4))
    print("✅ preparse_md_B_log.json created.")
    parse_dir(INPUT_DIR)