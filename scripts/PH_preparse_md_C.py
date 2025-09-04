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

INPUT_DIR = Path(r"../data/raw/md_C") ###### WILL BE FIREDATA IN BOX SUBFOLDER, (firedata/flammabilitydata/cone/Box/md_C)
OUTPUT_DIR_CSV = Path(r"../data/pre-parsed/Box/md_C") ###This will eventually be on firedata
METADATA_DIR = Path(r"../Metadata/preparsed/Box/md_C")###Store here for now, but will be on firedata either sep or together with csvs
LOG_FILE = Path(r"..") / "preparse_md_C_log.json"


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
        out_path = False
        if pct == 100:
            print(colorize(f"Parsed {path} successfully\n", "green"))
            files_parsed_fully += 1
        elif pct == 0:
            print(colorize(f"{path} could not be parsed", "red"))
            out_path = Path(str(path).replace('md_C', 'md_C_bad'))
        else:
            print(colorize(f'{pct}% of tests in {path} parsed succesfully\n', 'yellow'))
            files_parsed_partial += 1
            out_path = Path(str(path).replace('md_C', 'md_C_partial'))

        # If output path is set, ensure the directory exists and move
        if out_path:
            out_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(path, out_path)
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

def is_metadata_line(line):
    """Detect if a line contains metadata (date-number; e.g. 9/30/82-198)."""
    return re.search(r'\d{1,2}/\d{1,2}/\d{2}-(\d{1,4})', line)

def get_tests(lines):
    """
    Returns: dict {test_key: [lines]}
    * Preamble lines before first test are added after first test's metadata.
    * Keys are 'testXXX' where XXX is the number after the hyphen.
    """
    tests = {}
    current_test_lines = []
    preamble_lines = []
    metadata_buffer = []
    test_key = None
    found_first_test = False

    for line in lines:
        meta_match = is_metadata_line(line)
        if meta_match:
            # Save previous test
            if test_key is not None:
                # Attach buffered metadata
                block_lines = metadata_buffer + current_test_lines
                tests[test_key] = block_lines
            # Start new test: key = testXXX
            number = meta_match.group(1)
            test_key = f"test{number}"
            metadata_buffer = [line]
            current_test_lines = []
            if not found_first_test and preamble_lines:
                # Insert preamble lines right after metadata
                metadata_buffer += preamble_lines
                preamble_lines = []  # clear preamble
            found_first_test = True
        else:
            if not found_first_test:
                preamble_lines.append(line)
            else:
                current_test_lines.append(line)
    # Save last test
    if test_key is not None:
        block_lines = metadata_buffer + current_test_lines
        tests[test_key] = block_lines
    elif preamble_lines:
        tests['UNLABELED'] = preamble_lines
    print(tests.keys())
    return tests
    

####### separate metadata from test data #######
#region get_data
def get_data(data):
    # data: list of lines (test)
    dataStart = -1
    dataEnd = len(data)
    massWStart = -1
    prevline = " "
    # Find the start and end lines for the time-series table
    for index, line in enumerate(data):
        if not line.strip():
            continue #get rid of whitespace lines
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

        # Mark ending of test data
        if (index == len(data)):
            dataEnd = index
            break
        prevline = uline
    test_data = data[dataStart:dataEnd]
    #print(f"{dataStart} to {dataEnd}")
    metadata = data[:dataStart] + data[dataEnd:]
    print(f"{dataStart} to {dataEnd}")
    filtered_test_data = []
    
    for line in test_data:
        uline = line.upper().strip()
        # Remove Page Headers if they have in table
        if any(bad in uline for bad in ('TEST', 'PAGE', 'HOR', 'VERT')):
            continue
        # Remove markdown delimiter rows like |---|---|---|...| or just ---... or lines with only pipes/spaces/hyphens
        if (line.strip().replace('-', '').replace('|', '').replace(' ', '') == '') \
           and ('-' in line or '|' in line):
            continue
        #Remove only unit rows
        if any(unit in uline for unit in ("S", "KG","M2","KW","KJ")) and not any(header in line for header in ("TIME", "H", "SUM", "MASS","CO", "AREA")):
            continue
        #Remove False Headers 
        if any(bad in uline for bad in ('INDEX', 'VALUE', 'COLUMN', "YEAR", "PRESSURE", "WIND", "TEMPERATURE")):
            continue
        # Optionally: remove lines that are only spaces
        if not uline.strip():
            continue
        filtered_test_data.append(line)
    #print(filtered_test_data[0])
    
    #print(f"HEADER ({len(filtered_test_data[0].split('|'))} cols):", filtered_test_data[0])
    #for i, row in enumerate(filtered_test_data[1:6]):
    #    print(f"ROW {i} ({len(row.split('|'))} cols):", row)
        # convert test_data to df
        pd_format_test_data = StringIO("\n".join(filtered_test_data))
    
    #Majority of tests pipe delimited, but some are multispace delimited    
    test_data_df = pd.read_csv(pd_format_test_data, sep="|", header= 0, engine = 'python')
    return test_data_df, metadata

# outputting dataframe to csv file
#region parse_data
def parse_data(data_df,test,file_name):
    data_df = data_df.iloc[:, 1:-1]

  
    
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
    #logfile.update({
    #        str(test_name) : f"{column_uniform} || #Col = {data_df.shape[1]}"
     #   })
    
    
    #Renaming Column Headers
    for i, column in enumerate(data_df.columns):
        if "TIME" in column:
            data_df.columns.values[i] = "Time (s)"
        elif ("Q-DOT" in column and "SUM" not in column) or ("HEAT RELEASE" in column and "CUMULATIVE" not in column) :
            data_df.columns.values[i] = "HRRPUA (kW/m2)"
        elif "SUM" in column or "CUMULATIVE" in column:
            data_df.columns.values[i] = "Sum Q (MJ/m2)"
        elif "MASS" in column:
            # Additional check: Is this column monotonically decreasing?
            col_vals = pd.to_numeric(data_df[column], errors='coerce').dropna()
            # Check monotonic decreasing
            if (col_vals.diff().dropna() <= 0).all() and (col_vals != 0.0).all():
                data_df.columns.values[i] = "Mass (kg)"
            else:
                data_df.columns.values[i] = "MLR (g/s)"
        elif "COMB" in column or "EFFECTIVE" in column or "HEAT OF" in column:
            data_df.columns.values[i] = "HT Comb (MJ/kg)"
        elif "CO2" in column or "C02" in column:
            data_df.columns.values[i] = "CO2 (kg/kg)"
        elif ("CO" in column or "C0" in column) and "2" not in column:
            data_df.columns.values[i] = "CO (kg/kg)"
        elif "AIR" in column:
            data_df.columns.values[i] = "Air/Sample (kg/kg)"
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
    metadata = None
    metadata_json["Comments"] = []
    #First item in metadata being used to get info we can get, parse string, also add to comments incase something doesnt parse SmURF can fix
    #all subsequent, mostly useless, appeneded to comments
    METADATA_DIR.mkdir(parents=True, exist_ok=True)

    # checking for existing test metadata file 
    #if meta_path.exists():
     #   with open(meta_path, "r", encoding="utf-8") as w:  
      #      metadata_json = json.load(w)

    for i,line in enumerate(input):
        if i == 0:
            metadata = line
            metadata_json["Comments"].append(line)
        elif (line.strip().replace('-', '').replace('|', '').replace(' ', '').replace(r'\n','') != ''):
            metadata_json["Comments"].append(line)

    # metadata = list of metadata blocks as str
    #print(metadata)

    
    ############ finding metadata fields ############
    metadata_json["Material ID"] = None
    orient_idx = None
    slash_idx = None
    if "HOR" in metadata:
        orient_idx = metadata.find("HOR")
        metadata_json["Orientation"] = "HORIZONTAL"
    elif "VERT" in metadata:
        orient_idx = metadata.find("VERT")
        metadata_json["Orientation"] = "VERTICAL"
    if orient_idx:
        metadata_json["Material Name"] = metadata[:orient_idx-1]
    date_testnum =re.search(r'\d{1,2}/\d{1,2}/\d{2}-(\d{1,4})', metadata)
    dateidx = date_testnum.start()
    if date_testnum:
        metadata_json["Test Date"] = date_testnum.group(0).split("-")[0]
        metadata_json['Specimen Number'] = int(date_testnum.group(0).split("-")[1])
    if "/M2" in metadata:
        slash_idx = metadata.find("/M2")
        flux_str = metadata[slash_idx - 6: slash_idx + 3]
        metadata_json["Heat Flux (kW/m2)"] = get_number(flux_str,"int")
    elif "/CM2" in metadata:
        slash_idx = metadata.find("/CM2")
        flux_str = metadata[slash_idx - 6: slash_idx + 3]
        small_flux = get_number(flux_str, "flt")
        metadata_json["Heat Flux (kW/m2)"] = int(small_flux * 10)
    potential_mass_str =  metadata[slash_idx +4: dateidx]
    mass = get_number(potential_mass_str, "flt")
    if mass == None:
        mass_match = re.search(r'\((\d+(?:\.\d+)?)', metadata[:orient_idx])
        if mass_match:
            mass = get_number(mass_match.group(1), "flt")
    metadata_json["Sample Mass (g)"] = mass
        

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
    metadata_json["Original Source"] = "Box/md_C"
    metadata_json['Data Corrections'] =[]
    #update respective test metadata file
    with open(meta_path, "w", encoding="utf-8") as f:
        f.write(json.dumps(metadata_json, indent=4)) 
    print(colorize(f"Generated {meta_path}", "blue"))


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
    logfile = {}
    with open(LOG_FILE, "w", encoding="utf-8") as f:
        f.write(json.dumps(logfile, indent=4))
    print("✅ preparse_md_C_log.json created.")
    parse_dir(INPUT_DIR)