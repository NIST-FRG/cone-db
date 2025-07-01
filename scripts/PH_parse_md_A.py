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

INPUT_DIR = Path(r"../data/raw/md_A")
OUTPUT_DIR_CSV = Path(r"../data/auto-processed/md_A")
LOG_FILE = INPUT_DIR / "md_A_log.json"

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
                    str(path.name)[0:8:1] : str(e)
                })
            with open(LOG_FILE, "w", encoding="utf-8") as f:
	            f.write(json.dumps(logfile, indent=4))

            print(colorize(f" - Error parsing {path}: {e}\n", "red"))
            continue
        print(colorize(f"Parsed {path} successfully\n", "green"))
        files_parsed_successfully += 1
    
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
    tests = get_tests(lines)

    for test in tests:
        # for each test, separate data from metadata
        test_data_df, metadata = get_data(tests[test])
        # generate test data csv
        logfile = parse_data(test_data_df,test,file_path.name)
        #update md_A_log.json
        with open(LOG_FILE, "w", encoding="utf-8") as f:
	        f.write(json.dumps(logfile, indent=4))

   

####### separate tests in file #######
#region get_tests
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
        
        # ensure test number exists and doesn't equal the previous test number
        if (test_number_match is not None) and (line[test_number_match.start():] != test_number):
            # assumes test number will be end of string, gets start of match to EOL
            test_number = line[test_number_match.start():test_number_match.end()]
        if test_number != -1:
            if test_number in tests:
                tests[test_number].append(line)
            else:
                tests[test_number] = [line]
        
    print(tests.keys())

    return tests    
    

####### separate metadata from test data #######
#region get_data
def get_data(data):
    # data  = list of lines (test)
    
    dataStart = -1
    dataEnd = -1
    index = 0
    #has_page = False
    for line in data:
        line = str(line.upper())
        time_index = line.find("TIME")
        # if "times"
        if ("TIMES" in line):
            if dataStart == -1:
                 dataStart = index
            #has_page = True
        # if "time |"
        elif (time_index != -1):
            # check if | in vicinity
            for i in range(4,8):
                if (time_index+i < len(line)) and str(line[time_index+i]) == "|":
                    if dataStart == -1:
                        dataStart = index
                    #has_page = True
                    break                    
        # mark ending of test data
        #if ("---" == line or index == len(data)-1) and has_page:
        if (("PARAMETER SHEET" in line) or index == len(data)-1):
            #has_page = False
            dataEnd = index
            break
        index += 1

    test_data = data[dataStart:dataEnd]
    print(f"{dataStart} to {dataEnd}")
    metadata = data[:dataStart] + data[dataEnd:]
    
    # convert test_data to df
    pd_format_test_data = StringIO("\n".join(test_data))
    test_data_df = pd.read_csv(pd_format_test_data, sep="|")

    return test_data_df, metadata

# outputting dataframe to csv file
#region parse_data
def parse_data(data_df,test,file_name):
    data_df = data_df.iloc[:, 1:-1]

    # extract indices of separate datatables
    new_table_start = 0
    col_idx = data_df.columns[0]
    for index,row in data_df.iterrows():
        # new table starting where time is 0 again
        if (index != 1) and (str(row[col_idx]).strip() == '0.0'):
            # find column header row
            for i in range(1,5):
                first_col_cell = str(data_df.iloc[index-i,0])
                if "T" in first_col_cell.upper():
                    new_table_start = index-i
                    break
    
    # save new datatable as df
    new_table = data_df.iloc[new_table_start:,1:]
    # transform new table into additional columns
    for col in new_table.columns:
        #skip first row
        if pd.notna(new_table.iloc[0][col]):
            new_col_name = str(new_table.iloc[0][col]).strip()
            #init new column and fill
            data_df[new_col_name] = np.nan
            data_df[new_col_name] = data_df[new_col_name].astype("object")  # make string-compatible
            data_df.loc[0:(len(new_table)-2),new_col_name] = new_table.iloc[1:][col].values 

    # remove new table at original location
    data_df.iloc[new_table_start:,:] = np.nan
    
    def delete_cells(col):
        # Convert to string, strip whitespace
        col_stripped = col.astype(str).str.strip()

        # Define mask for valid (non-empty) values
        is_not_empty = col.notna() & ~col_stripped.isin(['', '-', ' - '])

        #Exclude any cells containing letters
        has_letters = col.astype(str).str.contains(r'[a-zA-Z]', na=False)

        # Exclude values with more than 2 dashes
        too_many_dashes = col_stripped.str.count('-') > 2

        # Final mask: valid and not full of dashes
        mask = is_not_empty & ~too_many_dashes & ~has_letters

        # Filter and shift
        non_empty = col[mask]
        n_missing = len(col) - len(non_empty)
        return pd.Series(list(non_empty) + [np.nan]*n_missing, index=col.index)

    # remove all miscellaneous cells
    data_df = data_df.apply(delete_cells)

    # remove unnecessary headers
    # data_df = data_df[~data_df.astype(str).apply(lambda row: row.str.contains("TIME", case=False, na=False).any(), axis=1)]

    '''
    # detect every header
    col_idx = data_df.columns[1]
    prev = data_df.columns[1]
    for index,row in data_df.iterrows():
        if any('TIME' in str(cell) for cell in row) and column_head != row[col_idx]:
    '''

    OUTPUT_DIR_CSV.mkdir(parents=True, exist_ok=True)

    #generate test data csv name
    test_name = test.casefold()
    test_name = test_name.replace(" ", "")
    test_name = test_name + "_" + file_name[0:8:1]
    test_name = f"{test_name}.csv"

    #generating contents for md_A_log
    with open(LOG_FILE, "r", encoding="utf-8") as w:  
        logfile = json.load(w)

    # checking validity of data parsing
    data_df_cols = data_df.iloc[:,:-1]
    column_counts = data_df_cols.count()
    if column_counts.nunique() != 1:
        column_uniform = "Datatable columns are not uniform"
    else:
        column_uniform = "Datatable columns are uniform"
    # update md_A_log based off uniformity of columns
    logfile.update({
            str(test_name) + "_cols" : f"{column_uniform} || #Col = {data_df.shape[1]}"
        })
    
    # renaming column headers
    if data_df.shape[1] == 12:
        data_df.columns = ['Time (s)', 'Q-Dot (kW/m2)', 'Sum Q (MJ/m2)', 'M-Dot (g/s-m2)', 'Mass Loss (kg/m2)', 'HT Comb (MJ/kg)', 'Ex Area (m2/kg)', 'CO2% (kg/kg)', 'CO% (kg/kg)', 'H2O% (kg/kg)', 'H\'carbs% (kg/kg)', 'HCl% (kg/kg)']
    
    # replacing "*" with NaN
    # data_df = data_df.apply(lambda col: col.map(lambda x: np.nan if "*" in str(x) else x))

    output_path = OUTPUT_DIR_CSV / test_name
    data_df.to_csv(output_path, index=False)
    print(colorize(f"Generated {output_path}", "blue"))

    return logfile




#region main
if __name__ == "__main__":
    # if log file doesn't exist, create
    #if not os.path.exists("../data/raw/md_A/md_A_log.json"):
    logfile = {}
    with open(LOG_FILE, "w", encoding="utf-8") as f:
        f.write(json.dumps(logfile, indent=4))
    print("✅ md_A_log.json created.")
    parse_dir(INPUT_DIR)