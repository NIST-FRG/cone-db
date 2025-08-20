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
import shutil
from utils import calculate_HRR, calculate_MFR, colorize

INPUT_DIR = Path(r"../data/raw/md_A")
OUTPUT_DIR_CSV = Path(r"../data/pre-parsed/md_A")
METADATA_DIR = Path(r"../metadata/md_A/preparsed")
LOG_FILE = Path(r"..") / "preparse_md_A_log.json"

'''
#Test pathing
INPUT_DIR = Path(r"../data/raw/md_A_more_columns")
OUTPUT_DIR_CSV = Path(r"../data/pre-parsed/md_A_more_columns")
METADATA_DIR = Path(r"../metadata/md_A_more_columns/preparsed")
LOG_FILE = Path(r"..") / "preparse_md_A_more_log.json"
'''


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
        pct = parse_file(path)
        if pct == 100:
            print(colorize(f"Parsed {path} successfully\n", "green"))
            files_parsed_fully += 1
        elif pct == 0:
            print(colorize(f"{path} could not be parsed", "red"))
        else:
            print(colorize(f'{pct}% of tests in {path} parsed succesfully\n', 'yellow'))
            files_parsed_partial += 1
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
    tests = get_tests(lines)
    numtests = len(tests)
    parsed = 0
    for i, test in enumerate(tests):
        try:
            # for each test, separate data from metadata
            test_data_df, metadata = get_data(tests[test])
            # generate test data csv
            logfile,test_name = parse_data(test_data_df,test,file_path.name)
            # parse through and generate metadata json file
            logfile = parse_metadata(metadata,test_name,logfile)
            #update md_A_log.json
            with open(LOG_FILE, "w", encoding="utf-8") as f:
                f.write(json.dumps(logfile, indent=4))
            parsed += 1
        except Exception as e:
            # log error in md_A_log
            with open(LOG_FILE, "r", encoding="utf-8") as w:  
                logfile = json.load(w)
            logfile.update({
                    str(test) : str(e)
                })
            with open(LOG_FILE, "w", encoding="utf-8") as f:
	            f.write(json.dumps(logfile, indent=4))

            print(colorize(f" - Error parsing {test}: {e}\n", "red"))
            continue
    pct_parsed = (parsed / numtests) *100
    return pct_parsed
   

####### separate tests in file #######
#region get_tests
# splits file in list of tests, stores as {tests} <key=test_number>
def get_tests(file_contents):
    test_number = -1
    tests = {}
    # Use while loop for lookahead and test detection
    for line in file_contents:
        line = str(line).upper().strip()
        test_number_match = re.search(r"TEST\s+\d{4}", line)

        if test_number_match is not None:
            # Extract the matched substring (could be e.g. "TEST 2865", "TEST   2865")
            test_number_str = line[test_number_match.start():test_number_match.end()]
            # Only one space btwn test and #:
            real_test_number = re.sub(r"\s+", " ", test_number_str)
            # Only update if changed:
            if real_test_number != test_number:
                test_number = real_test_number

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
    # data: list of lines (test)
    dataStart = -1
    dataEnd = -1
    massWStart = -1

    # Find the start and end lines for the time-series table
    for index, line in enumerate(data):
        # Normalize line for easy matching
        uline = line.upper().strip()
        # Only match time-series table headers, not summary tables
        # Require TIME and at least one key column typical of measurements
        if (
            (uline.startswith("|TIME") or uline.startswith("| TIME") or uline.startswith("TIME")) and
            any(key in uline for key in ("SUM", "DOT", "H"))
        ):
            if dataStart == -1:
                dataStart = index

        # Find end of starting metadata chunk
        if massWStart == -1 and ("MASS WEIGHTED" in uline):
            massWStart = index

        # Mark ending of test data
        if ("PARAMETER SHEET" in uline) or (index == len(data)-1):
            dataEnd = index
            break

    test_data = data[dataStart:dataEnd]
    #print(f"{dataStart} to {dataEnd}")
    metadata = data[:massWStart] + data[dataEnd:]
    print(f"{dataStart} to {dataEnd}")
    filtered_test_data = []
    
    for line in test_data:
        # Remove Page Headers if they have in table
        if any(bad in line for bad in ('TEST', 'PAGE', 'HOR', 'VERT')):
            continue
        # Remove markdown delimiter rows like |---|---|---|...| or just ---... or lines with only pipes/spaces/hyphens
        if (line.strip().replace('-', '').replace('|', '').replace(' ', '') == '') \
           and ('-' in line or '|' in line):
            continue
        #Remove only unit rows
        if any(unit in line for unit in ("S", "KG","M2","KW","KJ")) and not any(header in line for header in ("TIME","DOT", "H", "SUM", "MASS","CO", "AREA")):
            continue
        # Optionally: remove lines that are only spaces
        if not line.strip():
            continue
        filtered_test_data.append(line)
    #print(filtered_test_data[0])
    
    #print(f"HEADER ({len(filtered_test_data[0].split('|'))} cols):", filtered_test_data[0])
    #for i, row in enumerate(filtered_test_data[1:6]):
     #   print(f"ROW {i} ({len(row.split('|'))} cols):", row)
        # convert test_data to df
        pd_format_test_data = StringIO("\n".join(filtered_test_data))
    
    #Majority of tests pipe delimited, but some are multispace delimited    
    test_data_df = pd.read_csv(pd_format_test_data, sep="|", header= 0, engine = 'python')
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
    test_filename = test_name + "_" + file_name[0:8:1]
    test_name = f"{test_filename}.csv"

    #generating contents for md_A_log
    with open(LOG_FILE, "r", encoding="utf-8") as w:  
        logfile = json.load(w)
    #print(data_df)
    # checking validity of data parsing
    data_df_cols = data_df.iloc[:,:-1]
    column_counts = data_df_cols.count()
    if column_counts.nunique() != 1:
        column_uniform = "Datatable columns are not uniform"
    else:
        column_uniform = "Datatable columns are uniform"
    # update md_A_log based off uniformity of columns
    logfile.update({
            str(test_name) : f"{column_uniform} || #Col = {data_df.shape[1]}"
        })
    
    # renaming column headers
    #if data_df.shape[1] == 12:
    #    data_df.columns = ['Time (s)', 'Q-Dot (kW/m2)', 'Sum Q (MJ/m2)', 'M-Dot (g/s-m2)', 'Mass Loss (kg/m2)', 
    # 'HT Comb (MJ/kg)', 'Ex Area (m2/kg)', 'CO2 (kg/kg)', 'CO (kg/kg)', 'H2O (kg/kg)', 'H\'carbs (kg/kg)', 'HCl (kg/kg)']
    
    for i, column in enumerate(data_df.columns):
        if "TIME" in column:
            data_df.columns.values[i] = "Time (s)"
        elif "Q-DOT" in column:
            data_df.columns.values[i] = "Q-Dot (kW/m2)"
        elif "SUM Q" in column:
            data_df.columns.values[i] = "Sum Q (MJ/m2)"
        elif "M-DOT" in column:
            data_df.columns.values[i] = "M-Dot (g/s-m2)"
        elif "MASS" in column and "LOSS" in column:
            if "M-Dot (g/s-m2)" not in data_df.columns.values:
                data_df.columns.values[i] = "M-Dot (g/s-m2)"
                #some tests (ex 2227) have m-dot labled as mass loss, no cumulative mass loss stored so this should correct
                #if this becomes an issue (Mass loss listed before MLR) can switch to check if monotonically inc
            else:
                data_df.columns.values[i] = "Mass Loss (kg/m2)"
        elif "HT" in column:
            data_df.columns.values[i] = "HT Comb (MJ/kg)"
        elif "EX" in column:
            data_df.columns.values[i] = "Ex Area (m2/kg)"
        elif "CO2" in column or "C02" in column:
            data_df.columns.values[i] = "CO2 (kg/kg)"
        elif ("CO" in column or "C0" in column) and "2" not in column:
            data_df.columns.values[i] = "CO (kg/kg)"
        elif "H2" in column:
            #some of the O were seen as 0, H2 to remove error
            data_df.columns.values[i] = "H2O (kg/kg)"
        elif "CARBS" in column:
            data_df.columns.values[i] ="H'carbs (kg/kg)"
        elif "HCL" in column:
            data_df.columns.values[i] = "HCl (kg/kg)"
        else:
            msg = f'Illegal Column Detected: {column}'
            raise Exception(msg)


    # replacing "*" with NaN
    data_df = data_df.apply(lambda col: col.map(lambda x: np.nan if "*" in str(x) else x))

    output_path = OUTPUT_DIR_CSV / test_name
    data_df.to_csv(output_path, index=False)
    print(colorize(f"Generated {output_path}", "blue"))

    return logfile, test_filename


####### metadata clean and output functions #######
#region parse_metadata
# clean and output metadata as json
def parse_metadata(input,test_name,log_file):
    meta_filename = test_name + ".json"
    meta_path = METADATA_DIR / meta_filename
    metadata_json = {}
    metadata = []

    METADATA_DIR.mkdir(parents=True, exist_ok=True)

    # checking for existing test metadata file 
    if meta_path.exists():
        with open(meta_path, "r", encoding="utf-8") as w:  
            metadata_json = json.load(w)

    for line in input:
        # Preprocess line to remove excessive whitespace after '='
        line = re.sub(r'=\s*', '= ', line)  # Replace '=' followed by whitespace with '= '
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
    metadata_json["comments"] = []
    prev_item = None
    metadata_json["material_id"] = None
    for item in metadata:
        #print(metadata.index(item),item)
        if metadata.index(item) == 0:
            metadata_json["laboratory"] = item
        elif "IRRADIANCE" in item:
            metadata_json["heat_flux_kW/m2"] = get_number(item,"int")
        elif "TEST" in prev_item and not metadata_json.get("material_name"):
            metadata_json["material_id"] = None
            metadata_json["material_name"] = item    
        elif "HOR" in item:
            metadata_json["orientation"] = "HORIZONTAL"
        elif "VERT" in item:
            metadata_json["orientation"] = "VERTICAL"
        elif "CALIBRATION" in item:
            metadata_json["c_factor"] = get_number(item[3:],"flt")
        elif "INITIAL MASS=" in item:
            metadata_json["initial_mass_g"] = get_number(item[3:],"flt")
        elif "FINAL MASS=" in item:
            metadata_json["final_mass_g"] = get_number(item[3:],"flt")
        elif "SURFACE AREA" in item:
            metadata_json["surface_area_m2"] = get_number(item[3:],"flt")
        elif "SOOT AVERAGE" in item:
            metadata_json["soot_average_g/g"] = get_number(item,"exp")
        elif "MASS CONSUMED" in item:
            metadata_json["mass_consumed"] = get_field(item)
        elif item.find("CONVERSION FACTOR") == 0:
            metadata_json["conversion_factor"] = get_field(item)
        elif "TIME TO IGNITION" in item:
            metadata_json["t_ign_s"] = get_number(item,"int")
        elif "PEAK Q-DOT" in item:
            match = re.search(r'(\d+)\s+KW', item)
            if match:
                metadata_json["peak_q_dot_kw/m2"] = int(match.group(1))
        elif "PEAK M-DOT" in item:
            match = re.search(r'(\d+\.\d+)\s+G', item)
            if match:
                metadata_json["peak_m_dot_g/s-m2"] = float(match.group(1))
        elif "TEST" in item:
            match = re.search(r'TEST\s+(\d{4})', item)
            if match:
                metadata_json["specimen_number"] = int(match.group(1))
        elif re.search(r'\d+\s+[A-Z]{3}\s+\d{4}', item) is not None:
            metadata_json["date"] = item
        else:
            metadata_json["comments"].append(item)
        prev_item = item

    metadata_json["number_of_fields"] = len(metadata_json)

    #autoprocessed values
    if ("initial_mass_g" in metadata_json) and ("final_mass_g" in metadata_json):
        metadata_json["mf/m0_g/g"] = float(metadata_json["final_mass_g"]) / float(metadata_json["initial_mass_g"])

    #update respective test metadata file
    with open(meta_path, "w", encoding="utf-8") as f:
        f.write(json.dumps(metadata_json, indent=4))

    # adding field count to test in log file(md_A_log.json)
    fields_found = len(metadata_json)
    log_test = str(test_name) + ".csv"
    log_file.update({
            log_test : log_file[log_test] + " || #Metadata_fields = " + str(fields_found)
        })

    print(colorize(f"Generated {meta_path}", "blue"))

    return log_file

    

    # metadata table checker
    # if line[0] == | then until not | at line[0] add to a table


    # # Determine output path
    # Path(OUTPUT_DIR_JSON / str(test_year)).mkdir(parents=True, exist_ok=True)

    # data_output_path = Path(OUTPUT_DIR_JSON) / str(test_year) /f"{Path(file_path).stem}.csv"
    # metadata_output_path = Path(OUTPUT_DIR_JSON) / str(test_year) / f"{Path(file_path).stem}.json"

#region helpers
#get number(int,float,exponent,)
def get_number(item, num_type):
    number = "Not found"
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
    logfile = {}
    with open(LOG_FILE, "w", encoding="utf-8") as f:
        f.write(json.dumps(logfile, indent=4))
    print("✅ preparse_md_A_log.json created.")
    parse_dir(INPUT_DIR)