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


INPUT_DIR = Path(r"../data/raw/md_B")
OUTPUT_DIR_CSV = Path(r"../data/pre-parsed/md_B")
METADATA_DIR = Path(r"../Metadata/preparsed/md_B")###Store here for now, but will be on firedata either sep or together with csvs
LOG_FILE = Path(r"..") / "preparse_md_B_log.json"


'''
#Test pathing
INPUT_DIR = Path(r"../data/raw/test")
OUTPUT_DIR_CSV = Path(r"../data/pre-parsed/test")
METADATA_DIR = Path(r"../metadata/test/preparsed")
LOG_FILE = Path(r"..") / "test.json"
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
        elif pct == 0:
            print(colorize(f"{path} could not be parsed", "red"))
            out_path = Path(str(path).replace('md_B', 'md_B_bad'))
        else:
            print(colorize(f'{pct}% of tests in {path} parsed succesfully\n', 'yellow'))
            files_parsed_partial += 1
            out_path = Path(str(path).replace('md_B', 'md_B_partial'))

        # If output path is set, ensure the directory exists and copy
        #if out_path:
        #    out_path.parent.mkdir(parents=True, exist_ok=True)
        #    shutil.move(path, out_path)
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
    for test in tests:
        try:
            # for each test, separate data from metadata
            test_data_df, metadata = get_data(tests[test])
            # generate test data csv
            data_df,test_filename = parse_data(test_data_df,test,file_path.name)
            # parse through and generate metadata json file
            parse_metadata(metadata,test_filename)
            test_name = f"{test_filename}.csv"
            output_path = OUTPUT_DIR_CSV / test_name
            data_df.to_csv(output_path, index=False)
            print(colorize(f"Generated {output_path}", "blue"))
            parsed += 1
        except Exception as e:
            # log error in md_A_log
            with open(LOG_FILE, "r", encoding="utf-8") as w:  
                logfile = json.load(w)
            logfile.update({
                f"{file_path.name}-{test}": str(e)
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
    for i in range(len(file_contents) - 2):  # Stop at len - 2 to allow i+2 access
        line = str(file_contents[i]).upper().strip()
        test_match = re.search(r"\((\d{3,4})\)", line)
        if test_match is not None:
            line_after_two = file_contents[i + 2].upper().strip()

            max_test = re.search(r"MAX", line_after_two)
            param_test = re.search(r"PARAMETER", line_after_two)
            
            # ensure this is a new test
            if (max_test is not None) or (param_test is not None):
                test_number = f"Test {test_match.group(1)}"
                # print(f"Match on line {i}: {line}")

        # adding lines to respective test/key
        if test_number != -1:
            if test_number in tests:
                tests[test_number].append(line)
            else:
                tests[test_number] = [line]

    # add skipped last two lines to the last detected test
    tests[test_number].append(str(file_contents[len(file_contents) - 2]).upper().strip())
    tests[test_number].append(str(file_contents[len(file_contents) - 1]).upper().strip())

    print(tests.keys())

    return tests    
    

####### separate metadata from test data #######
#region get_data
#region get_data
def get_data(data):
    # data  = list of lines (test)
    
    dataStart = -1
    dataEnd = -1
    index = 0
    #has_page = False
    for i, line in enumerate(data):
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
            max_test = re.search(r"MAX", line)
            param_test = re.search(r"PARAMETER", line)
            for i in range(1,10):
                if (time_index+i < len(line)) and str(line[time_index+i]) == "|" and (max_test is None) and (param_test is None):
                    if dataStart == -1:
                        dataStart = index
                    #has_page = True
                    break                    
    
        index += 1

    test_data = data[dataStart:]
    filtered_test_data = []
    
    for line in test_data:
        if any(bad in line for bad in ('TEST', 'PAGE', 'HOR', 'VERT', "MO=", "MF=", "IGN")):
            continue
        # Remove markdown delimiter rows like |---|---|---|...| or just ---... or lines with only pipes/spaces/hyphens
        if (line.strip().replace('-', '').replace('|', '').replace(' ', '') == '') \
           and ('-' in line or '|' in line):
            continue
        #Remove only unit rows
        if any(unit in line for unit in ("S", "SEC","KG","M2","M3","KW","KJ")) and not any(header in line for header in ("TIME","DOT", "H", "SUM", "MASS","CO", "AREA", "DUCT")):
            continue
        # Optionally: remove lines that are only spaces
        if not line.strip():
            continue
        filtered_test_data.append(line)
    #for i, row in enumerate(filtered_test_data):
     #   print(f"ROW {i} ({len(row.split('|'))} cols):", row)


    #print(f"{dataStart} to {dataEnd}")
    metadata = data[:dataStart] + data[dataEnd:]
    print(f"Data Table from {dataStart} to {dataEnd}")
    
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
    table_idx_list = []
    for index,row in data_df.iterrows():
        # new table starting where time is 0 again
        if (index != 1) and (str(row[col_idx]).strip() == '0.'):
            # find column header row
            for i in range(1,5):
                first_col_cell = str(data_df.iloc[index-i,0])
                if "T" in first_col_cell.upper():
                    new_table_start = index-i
                    table_idx_list.append(index-i)
                    break

    for idx in range(len(table_idx_list)):
        # save new datatable as df
        if idx == (len(table_idx_list)-1):
            new_table = data_df.iloc[table_idx_list[idx]:,1:]
        else:
            new_table = data_df.iloc[table_idx_list[idx]:table_idx_list[idx+1],1:]
        # transform new table into additional columns
        for col in new_table.columns:
            #skip first row
            if pd.notna(new_table.iloc[0][col]):
                new_col_name = str(new_table.iloc[0][col]).strip()
                #init new column and fill
                data_df[new_col_name] = np.nan
                data_df[new_col_name] = data_df[new_col_name].astype("object")  # make string-compatible
                data_df.loc[0:(len(new_table)-2),new_col_name] = new_table.iloc[1:][col].values 

    # remove new table(s) at original location
    data_df.iloc[table_idx_list[0]:,:] = np.nan
    

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

    # checking validity of data parsing
    data_df_cols = data_df.iloc[:,:-1]
    column_counts = data_df_cols.count()
    if column_counts.nunique() != 1:
        column_uniform = "Datatable columns are not uniform"
    else:
        column_uniform = "Datatable columns are uniform"
    # update md_A_log based off uniformity of columns
    #logfile.update({
   #         str(test_name) : f"{column_uniform} || #Col = {data_df.shape[1]}"
    #    })


     #Renaming Column Headers
    for i, column in enumerate(data_df.columns):
        if "TIME" in column:
            data_df.columns.values[i] = "Time (s)"
        elif "Q-DOT" in column:
            data_df.columns.values[i] = "Q-Dot (kW/m2)"
        elif "SUM Q" in column:
            data_df.columns.values[i] = "Sum Q (MJ/m2)"
        elif "MASS" in column:
            data_df.columns.values[i] = "Mass (g)"
        elif "MLOSS" in str(column.replace(" ", "")):
            if "M-Dot (g/s-m2)" not in data_df.columns.values:
                data_df.columns.values[i] = "MLR (g/s)"
        elif "COMB" in column:
            data_df.columns.values[i] = "HT Comb (MJ/kg)"
        elif "CO2" in column or "C02" in column:
            data_df.columns.values[i] = "CO2 (kg/kg)"
        elif ("CO" in column or "C0" in column) and "2" not in column:
            data_df.columns.values[i] = "CO (kg/kg)"
        elif "H2" in column:
            #some of the O were seen as 0, H2 to remove error
            data_df.columns.values[i] = "H2O (kg/kg)"
        elif "CARB" in column:
            data_df.columns.values[i] ="H'carbs (kg/kg)"
        elif "HCL" in column:
            data_df.columns.values[i] = "HCl (kg/kg)"
        elif "M-DUCT" in column:
            data_df.columns.values[i] = "M-Duct (kg/s)"
        elif "V-DUCT" in column: 
            data_df.columns.values[i] = "V-Duct (m3/s)"
        elif "SOOT" in column:
            data_df.columns.values[i] = "Soot (kg/kg)"
        elif "AREA" in column and "SUM" not in column:
            data_df.columns.values[i] = "Ex Area (m2/kg)"
        elif ("AREA" in column and "SUM" in column) or ("TOTALSMOKE" in str(column.replace(" ", ""))):
            data_df.columns.values[i] = "Sum Ex Area (m2/kg)"
        elif "SAMPTEMP" in str(column.replace(" ", "")):
            data_df.columns.values[i] = "Sample Temperature (Deg C)"
        else:
            msg = f'Illegal Column Detected: {column}'
            raise Exception(msg)
        
    # replacing "*" with NaN
    data_df = data_df.apply(lambda col: col.map(lambda x: np.nan if "*" in str(x) else x))

    return data_df, test_filename


####### metadata clean and output functions #######
#region parse_metadata
# clean and output metadata as json
def parse_metadata(input,test_name):
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
    metadata_json["Comments"] = []
    metadata_json["Specimen Number"] = str(test_name).split("_")[0].split("t")[-1] # just the number
    metadata_json["Material ID"] = None
    for item in metadata:
        if metadata.index(item) == 0:
            name = item.split("(", 1)[0].strip()
            metadata_json["Material Name"] = name
        if "Heat Flux (kW/m2)" not in metadata_json:
            if "KW/M2" in item:
                match = re.search(r'(\d+\s*KW/M2)', item)
            if match:
                substring = match.group(1)
            else:
                # Alternative: get all characters (digits, possibly units and spaces) just before KW/M2
                match = re.search(r'([^\s]+(?:\s*KW/M2))', item)
                substring = match.group(1) if match else None
            metadata_json["Heat Flux (kW/m2)"] = get_number(substring, "int")
        if "MAX TIME" in item:
            metadata_json["time_s"] = get_number(item, "flt")
        elif "MAX HEAT RELEASE" in item:
            metadata_json["Peak Heat Release Rate (kW/m2)"] = get_number(item, "flt")
        elif "HOR" in item:
            metadata_json["Orientation"] = "HORIZONTAL"
        elif "VERT" in item:
            metadata_json["Orientation"] = "VERTICAL"
        elif "MO" in item:
            metadata_json["Sample Mass (g)"]= get_number(item,"flt")
        elif "MF" in item:
            metadata_json["Residual Mass (g)"] = get_number(item,"flt")
        elif "T IGN" in item:
            metadata_json["Time to Ignition (s)"] = get_number(item,"int")
        elif re.search(r'\s*\d+\s+(([A-Z]{3})|([A-Z]{4}))\s+\d{2}', item) is not None:
            metadata_json["Test Date"] = str(item).strip()
        else:
            if "PAGE" not in item and "---" not in item:
                metadata_json["Comments"].append(item) 
        

    expected_keys = [
        "Institution",
        "Heat Flux (kW/m2)",
        "Material Name",
        "Orientation",
        "C Factor",
        "Sample Mass (g)",
        "Residual Mass (g)",
        "Surface Area (m2)",
        "Soot Average (g/g)",
        "Mass Consumed",
        "Conversion Factor",
        "Time to Ignition (s)",
        "Peak Heat Release Rate (kW/m2)",
        "Peak Mass Loss Rate (g/s-m2)",
        "Specimen Number",
        "Test Date",
        "Residue Yield (g/g)"
    ]
        #autoprocessed values
    if ("Sample Mass (g)" in metadata_json) and ("Residual Mass (g)" in metadata_json) and (metadata_json["Sample Mass (g)"] != None) and (metadata_json["Sample Mass (g)"] != None):
        metadata_json["Residue Yield (g/g)"] = float(metadata_json["Residual Mass (g)"]) / float(metadata_json["Sample Mass (g)"])
    for key in expected_keys:
        metadata_json.setdefault(key, None)
    metadata_json['Original Testname'] = test_name
    metadata_json ['Testname'] = None
    metadata_json['Instrument'] = "NBS Cone Calorimeter"
    metadata_json['Autoprocessed'] = None
    metadata_json['Preparsed'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    metadata_json['Parsed'] = None
    metadata_json['SmURF'] = None
    metadata_json['Bad Data'] = None
    metadata_json["Auto Prepared"] = None
    metadata_json["Manually Prepared"] = None
    metadata_json["Manually Reviewed Series"] = None
    metadata_json['Pass Review'] = None
    metadata_json["Published"] = None

    metadata_json['Data Corrections'] =[]

    #update respective test metadata file
    with open(meta_path, "w", encoding="utf-8") as f:
        f.write(json.dumps(metadata_json, indent=4))
    print(colorize(f"Generated {meta_path}", "blue"))

    

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
    print("✅ preparse_md_A_log.json created.")
    parse_dir(INPUT_DIR)