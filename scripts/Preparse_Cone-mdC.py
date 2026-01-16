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

#Path Handling: Relative to this script's location
SCRIPT_DIR = Path(__file__).resolve().parent         # .../coneDB/scripts
PROJECT_ROOT = SCRIPT_DIR.parent             # .../coneDB 

INPUT_DIR = PROJECT_ROOT / "data" / "raw" / "Box" / "md_C"### WILL BE FIREDATA IN BOX SUBFOLDER, (firedata/flammabilitydata/cone/Box/md_B)
OUTPUT_DIR = PROJECT_ROOT / "data" / "preparsed" / "Box" / "md_C"
LOG_FILE = PROJECT_ROOT / "preparse_md_C_log.json"


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
        elif pct == 0 or pct == None:
            print(colorize(f"{path} could not be parsed", "red"))
            out_path = Path(str(path).replace('md_C', 'md_C_bad'))
        else:
            print(colorize(f'{pct}% of tests in {path} parsed succesfully\n', 'yellow'))
            files_parsed_partial += 1
            out_path = Path(str(path).replace('md_C', 'md_C_partial'))

        # If output path is set, ensure the directory exists and move
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
                if "PH_preparse_md_C" in tb.filename and "get_number" not in tb.name:
                    fail = tb
                    break
            if not fail:
                print(tb_list)
                fail = tb_list[0] 
            location = f"{fail.filename.split("\\")[-1]}:{fail.lineno} ({fail.name})"
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
#region get_tests
# splits file in list of tests, stores as {tests} <key=test_number>
'''''
def get_tests(lines):
    """
    Returns: dict {test_key: [lines]}
    - Metadata marks the start of a test.
    - Gather all lines after metadata until next metadata or header line.
    - If a header line is hit, those and any following lines are 'preamble'
      for the next test (to be appended after its metadata).
    """
    tests = {}
    preamble = []
    test_key = None
    current_test_lines = []
    last_delim = None
    addtocurrent = False

    def is_table_header_line(line):
        return line.replace(" ", "").startswith("|TIME") and not any(
            bad in line.upper() for bad in ('INDEX', 'VALUE', 'COLUMN', "YEAR", "PRESSURE", "WIND", "TEMPERATURE"))

    for line in lines:
        print(line)
        meta_match = is_metadata_line(line)
        if meta_match:
           # print(line)
            if last_delim == None:
                print(last_delim)
                print("extmeta")
                # Metadata the first delimiter, start new test
                test_number = meta_match.group(1)
                test_key = f"test{test_number}"
                # Insert preamble if we have it
                current_test_lines = [line] + preamble
                preamble = []
                last_delim = 'ext_meta'
                addtocurrent = True
            elif last_delim == "ext_meta":
                #SHOULD NEVER HAPPEN
                pass
            elif last_delim == "int_meta" or last_delim == "int_head":
                #Previous delimter was internal metadata, so this is external for following test, order btwn tests flipped
                #Or previous delimiter was an internal header, so this is external for the following test, order between tests consistent
                print(last_delim)
                print("extmeta")
                tests[test_key] = current_test_lines
                test_number = meta_match.group(1)
                test_key = f"test{test_number}"
                # Insert preamble if we have it
                current_test_lines = [line] + preamble
                preamble = []
                last_delim = 'ext_meta'
                addtocurrent = True 
            elif last_delim == "ext_head":
                #Previous delimiter was an external header, so this is an internal metadata
                print(last_delim)
                print("intmeta")
                test_number = meta_match.group(1)
                test_key = f"test{test_number}"
                # Insert preamble if we have it
                current_test_lines = [line] + preamble
                preamble = []
                last_delim = 'int_meta'
                addtocurrent = True

           
        elif is_table_header_line(line):
           #print(line)
            if last_delim == None:
                print(last_delim)
                print("exthead")
                ###Data first
                preamble.append(line)
                addtocurrent = False
                last_delim = 'ext_head'
            elif last_delim == 'ext_meta':
                print(last_delim)
                print("inthead")
                #Previous delimiter was external metadata, so this is internal table header
                current_test_lines.append(line)
                last_delim = 'int_head'
            elif last_delim == 'int_meta' or last_delim =='int_head':
                print(last_delim)
                print("exthead")
                #Previous delimiter was internal metadata, so this is external header, same order btwn tests
                tests[test_key] = current_test_lines
                current_test_lines = []
                addtocurrent = False
                last_delim = 'ext_head'
                preamble.append(line)
            elif last_delim =='ext_head':
                #SHOULD NEVER HAPPEN
                pass
                 
        else:
            if addtocurrent:
                current_test_lines.append(line)
            else:
                preamble.append(line)

    # Save last test if any
    if test_key is not None and current_test_lines:
        tests[test_key] = current_test_lines
    elif preamble:
        # If something is left over and not assigned, call it unlabeled
        tests['UNLABELED'] = preamble
    print(tests.keys())
    return tests

    '''

def get_tests(lines):
    """
    Returns: dict {test_key: [lines]}
    - Metadata marks the start of a test.
    - Gather all lines after metadata until next metadata or header line.
    - If a header line is hit, those and any following lines are 'preamble'
      for the next test (to be appended after its metadata).
    """
    tests = {}
    preamble = []
    test_key = None
    current_test_lines = []
    last_delim = None
    addtocurrent = False
    def is_metadata_line(line):
        """
        Detect if a line contains metadata:
        - If "date-number" (e.g. 9/30/82-198): return (date, number)
        - If just "date" (e.g. 9/30/82): return (date, "unk#")
        - Else, return None
        """
        match = re.search(r'(\d{1,2}[/-]\d{1,2}[/-]\d{1,4})(?:-(\d{1,4}))?', line)
        return match
    def is_table_header_line(line):
        return line.replace(" ", "").startswith("|TIME") and not any(
            bad in line.upper() for bad in ('INDEX', 'VALUE', 'COLUMN', "YEAR", "PRESSURE", "WIND", "TEMPERATURE"))

    for line in lines:
        meta_match = is_metadata_line(line)

        if meta_match:
            if last_delim == None:
                # Metadata the first delimiter, start new test
                raw = meta_match.group(2)
                if raw == None:
                    test_number = "UNK"
                else:
                    test_number = raw.zfill(4)
                test_key = f"Test {test_number}"
                # Insert preamble if we have it
                current_test_lines = [line] + preamble
                preamble = []
                last_delim = 'ext_meta'
                addtocurrent = True
            elif last_delim == "ext_meta":
                #SHOULD NEVER HAPPEN UNLESS DUPLICATE
                pass
            elif last_delim == "int_meta" or last_delim == "int_head":
                #Previous delimter was internal metadata, so this is external for following test, order btwn tests flipped
                #Or previous delimiter was an internal header, so this is external for the following test, order between tests consistent
                tests[test_key] = current_test_lines
                raw= meta_match.group(2)
                if raw == None:
                    test_number = "UNK"
                else:
                    test_number = raw.zfill(4)
                test_key = f"Test {test_number}"
                # Insert preamble if we have it
                current_test_lines = [line] + preamble
                preamble = []
                last_delim = 'ext_meta'
                addtocurrent = True 
            elif last_delim == "ext_head":
                #Previous delimiter was an external header, so this is an internal metadata
                raw = meta_match.group(2)
                if raw == None:
                    test_number = "UNK"
                else:
                    test_number = raw.zfill(4)
                test_key = f"Test {test_number}"
                # Insert preamble if we have it
                current_test_lines = [line] + preamble
                preamble = []
                last_delim = 'int_meta'
                addtocurrent = True

           
        elif is_table_header_line(line):
           #print(line)
            if last_delim == None:
                ###Data first
                preamble.append(line)
                addtocurrent = False
                last_delim = 'ext_head'
            elif last_delim == 'ext_meta':
                #Previous delimiter was external metadata, so this is internal table header
                current_test_lines.append(line)
                last_delim = 'int_head'
            elif last_delim == 'int_meta' or last_delim =='int_head':
                #Previous delimiter was internal metadata, so this is external header, same order btwn tests
                tests[test_key] = current_test_lines
                current_test_lines = []
                addtocurrent = False
                last_delim = 'ext_head'
                preamble.append(line)
            elif last_delim =='ext_head':
                #SHOULD NEVER HAPPEN
                pass
                 
        else:
            if addtocurrent:
                current_test_lines.append(line)
            else:
                preamble.append(line)

    # Save last test if any
    if test_key is not None and current_test_lines:
        tests[test_key] = current_test_lines
    elif preamble:
        # If something is left over and not assigned, call it unlabeled
        tests['UNLABELED'] = preamble
    print(tests.keys())
    if 'UNLABELED' in tests.keys() or "UNK" in tests.keys():
        raise Exception("Likley typo in test numbers exist, please correct the markdown file.")
    return tests


####### separate metadata from test data #######
#region get_data
def get_data(data):
    # data: list of lines (test)
    dataStart = -1
    dataEnd = len(data)
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

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

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
            data_df.columns.values[i] = "THRPUA (MJ/m2)"
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
    data_df = data_df.apply(pd.to_numeric, errors = 'coerce').astype(float)
    data_df.columns = data_df.columns.astype(str) # make all column headers strings
    
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
        if len(data) > len(times):
            raise Exception(f"Column {c} exceeds the length of time in the test, please review markdown and pdf")
    return data_df, test_filename


####### metadata clean and output functions #######
#region parse_metadata
# clean and output metadata as json
def parse_metadata(input,test_name):
    meta_filename = test_name + ".json"
    meta_path = OUTPUT_DIR / meta_filename
    metadata_json = {}
    metadata = None
    comments = []
    #First item in metadata being used to get info we can get, parse string, also add to comments incase something doesnt parse SmURF can fix
    #all subsequent, mostly useless, appeneded to comments
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
    for i,line in enumerate(input):
        if i == 0:
            metadata = line
            comments.append(line)
        elif (line.strip().replace('-', '').replace('|', '').replace(' ', '').replace(r'\n','') != ''):
            comments.append(line)

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
    'Peak HRRPUA (kW/m2)', 'Peak MLRPUA Outlier',
    'Total Heat Release (MJ/m2)', 'Total Heat Release Outlier',
    'Average HoC (MJ/kg)', 'Average HoC Outlier',
    'Average Specific Extinction Area (m2/kg)', 'Average Specific Extinction Area Outlier',
    'Smoke Production Pre-ignition (m2/m2)','Smoke Production Pre-ignition Outlier',
    'Smoke Production Post-ignition (m2/m2)','Smoke Production Post-ignition Outlier',
    'Smoke Production Total (m2/m2)','Smoke Production Total Outlier',
    'Y_Soot (g/g)', 'Y_Soot Outlier',
    'Y_CO2 (g/g)', 'Y_CO2 Outlier',
    'Y_CO (g/g)', 'Y_CO Outlier',
    'Fire Growth Potential (m2/J)', 'Fire Growth Potential Outlier',
    'Ignition Energy (MJ/m2)', 'Ignition Energy Outlier',
    "t_flameout (s)","t_flameout outlier",
    'Comments', 'Data Corrections'
        ]

    for key in expected_keys:
        metadata_json.setdefault(key, None)
    metadata_json["Comments"] = comments
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
    
    date_testnum =re.search(r'(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})(?:-(\d{1,4}))?', metadata)
    dateidx = date_testnum.start()
    if date_testnum:     
        metadata_json["Test Date"] = date_testnum.group(1)        
        specnum =  date_testnum.group(2)             
        metadata_json['Specimen Number'] = specnum
    if "/M2" in metadata:
        slash_idx = metadata.find("/M2")
        flux_str = metadata[slash_idx - 6: slash_idx + 3]
        metadata_json["Heat Flux (kW/m2)"] = get_number(flux_str,"int")
    elif "/CM2" in metadata:
        slash_idx = metadata.find("/CM2")
        flux_str = metadata[slash_idx - 6: slash_idx + 3]
        small_flux = get_number(flux_str, "flt")
        metadata_json["Heat Flux (kW/m2)"] = int(small_flux * 10)
    elif "0FLUX" or "NOEXTERNALFLUX" in metadata.replace(" ", ""):
        slash_idx = metadata.find("UX")
        metadata_json["Heat Flux (kW/m2)"] = 0
    potential_mass_str =  metadata[slash_idx +4: dateidx]
    mass = get_number(potential_mass_str, "flt")
    if mass == None:
        mass_match = re.search(r'\((\d+(?:\.\d+)?)', metadata[:orient_idx])
        if mass_match:
            mass = get_number(mass_match.group(1), "flt")
    metadata_json["Sample Mass (g)"] = mass
        

    metadata_json['Original Testname'] = test_name
    metadata_json['Instrument'] = "NBS Cone Calorimeter"
    metadata_json['Preparsed'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    metadata_json["Original Source"] = "Box/md_C"
    metadata_json['Data Corrections'] =[]
    #update respective test metadata file
    with open(meta_path, "w", encoding="utf-8") as f:
        f.write(json.dumps(metadata_json, indent=4)) 
    print(colorize(f"Generated {meta_path}", "blue"))
    return None

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