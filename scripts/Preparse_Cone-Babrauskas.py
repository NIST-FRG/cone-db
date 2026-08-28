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

#HANDLING TABLE FORMAT: FOR NIST TABLE FILES
#Path Handling: Relative to this script's location
SCRIPT_DIR = Path(__file__).resolve().parent         # .../coneDB/scripts
PROJECT_ROOT = SCRIPT_DIR.parent             # .../coneDB 

INPUT_DIR = PROJECT_ROOT / "data" / "raw" / "Babrauskas"
OUTPUT_DIR = PROJECT_ROOT / "data" / "preparsed" / "Box" / "Babrauskas"
LOG_FILE = PROJECT_ROOT / "preparse_Babrauskas_log.json"

#region parse_dir
def parse_dir(input_dir):
    """Find and parse all .txt files in the input directory"""
    paths = Path(input_dir).glob("**/*.exp")
    paths = list(paths)
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
            out_path = Path(str(path).replace('Babrauskas', 'Babrauskas_bad'))
            
        out_path = False
        if pct == 100:
            print(colorize(f"Parsed {path} successfully\n", "green"))
            files_parsed_fully += 1
        elif pct == 0 or pct == None:
            print(colorize(f"{path} could not be parsed", "red"))
            out_path = Path(str(path).replace('Babrauskas', 'Babrauskas_bad'))
        else:
            print(colorize(f'{pct}% of file {path} parsed successfully\n', 'yellow'))
            files_parsed_partial += 1
            out_path = Path(str(path).replace('Babrauskas', 'Babrauskas_partial'))

        # If output path is set, ensure the directory exists and move
       # if out_path:
        #    out_path.parent.mkdir(parents=True, exist_ok=True)
        #    shutil.move(path, out_path)
            
    print(colorize(f"Files pre-parsed fully: {files_parsed_fully}/{files_parsed} ({((files_parsed_fully)/files_parsed) * 100:.1f}%)", "blue"))
    print(colorize(f"Files pre-parsed partially: {files_parsed_partial}/{files_parsed} ({((files_parsed_partial)/files_parsed) * 100:.1f}%)", "blue"))

#region parse_file
def parse_file(file_path):
    """Parse a single TABLE format file"""
    print(colorize(f"Parsing {file_path.name}:", "yellow"))
    
    try:
        # Try different encodings
        lines = None
        encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
        
        for encoding in encodings:
            try:
                with open(file_path, "r", encoding=encoding, errors='ignore') as file:
                    lines = file.readlines()
                    print(f"Read {len(lines)} lines from file using {encoding} encoding")
                    break
            except UnicodeDecodeError:
                continue
        
        if lines is None:
            print(colorize(f" - Could not decode {file_path.name} with any known encoding", "red"))
            with open(LOG_FILE, "r", encoding="utf-8") as w:  
                logfile = json.load(w)
            logfile.update({
                file_path.name: "Could not decode file with any encoding"
            })
            with open(LOG_FILE, "w", encoding="utf-8") as f:
                f.write(json.dumps(logfile, indent=4))
            return 0

        # Separate data from metadata
        test_data_df, metadata_dict = get_data(lines)
        
        # Check if we got valid data
        if test_data_df.empty:
            print(colorize(f" - No data found in {file_path.name}", "red"))
            with open(LOG_FILE, "r", encoding="utf-8") as w:  
                logfile = json.load(w)
            logfile.update({
                file_path.name: "No data found"
            })
            with open(LOG_FILE, "w", encoding="utf-8") as f:
                f.write(json.dumps(logfile, indent=4))
            return 0
        
        # Generate test data csv
        data_df, test_filename = parse_data(test_data_df, file_path.name)
        
        if data_df is None or data_df.empty:
            print(colorize(f" - Failed to parse data from {file_path.name}", "red"))
            with open(LOG_FILE, "r", encoding="utf-8") as w:  
                logfile = json.load(w)
            logfile.update({
                file_path.name: "Failed to parse data"
            })
            with open(LOG_FILE, "w", encoding="utf-8") as f:
                f.write(json.dumps(logfile, indent=4))
            return 0
        
        data_df = data_df.replace([np.inf, -np.inf], np.nan).dropna(how='all')
        
        test_name = f"{test_filename}.csv"
        output_path = OUTPUT_DIR / test_name
        
        if output_path.exists():
            old_df = pd.read_csv(output_path)
            # Compare old and new dataframes
            if old_df.equals(data_df):
                print(colorize(f"{test_filename} already exists and is identical. Skipping generation.", "blue"))
                return 100
            else:
                print(colorize(f"{test_filename} already exists but differs. Overwriting with new data.", "yellow"))
        
        # Parse through and generate metadata json file
        status = parse_metadata(metadata_dict, test_filename)
        
        if status == "SmURF":
            print(colorize(f"{test_filename} already SmURFed", "blue"))
            return 100
        elif status == "Bad":
            print(colorize(f"{test_filename} marked as bad data", "purple"))
            return 100
        
        # Write CSV file
        data_df.to_csv(output_path, index=False)
        print(colorize(f"Generated {output_path}", "blue"))
        return 100
            
    except Exception as e:
        tb_list = traceback.extract_tb(e.__traceback__)
        fail = None
        for tb in reversed(tb_list):
            if "Preparse_Cone" in tb.filename or "Parse_Cone" in tb.filename:
                if "get_number" not in tb.name:
                    fail = tb
                    break
        if not fail:
            print(colorize(f"Traceback: {tb_list}", "red"))
            fail = tb_list[-1] 
        location = f"{fail.filename.split('/')[-1]}:{fail.lineno} ({fail.name})"
        
        # Log error
        with open(LOG_FILE, "r", encoding="utf-8") as w:  
            logfile = json.load(w)
        logfile.update({
             file_path.name: f"{e} @ {location}"
        })
        with open(LOG_FILE, "w", encoding="utf-8") as f:
            f.write(json.dumps(logfile, indent=4))

        print(colorize(f" - Error parsing file: {e}", "red"))
        return 0  # Return 0 instead of raising

#region get_data
def get_data(file_path):
    """
    Separate metadata and vector data from TABLE format file.
    Parse VARIABLE sections to build dataframe with proper column headers including units.
    
    Args:
        file_path: Path to TABLE format file OR list of lines if already read
    
    Returns:
        tuple: (metadata_dict, data_df) where metadata_dict contains KEY VALUE pairs
               and data_df is a DataFrame with all vector data columns (headers include units)
    """
    metadata_dict = {}
    data_sections = []  # List of (column_name_with_units, data_values) tuples
    
    # Handle different input types
    if isinstance(file_path, list):
        # Already a list of lines
        lines = file_path
    else:
        # Need to read the file
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()
    
    # Find where vector data starts
    vector_data_start = None
    for i, line in enumerate(lines):
        if 'VECTOR DATA' in line.upper():
            vector_data_start = i + 1
            break
    
    if vector_data_start is None:
        print(colorize(f"No VECTOR DATA marker found", "red"))
        return metadata_dict, pd.DataFrame()
    
   # Parse metadata section (everything before VECTOR DATA)
    metadata_lines = lines[:vector_data_start - 1]
    key_counts = {}  # Track how many times each key has appeared
    i = 0

    while i < len(metadata_lines):
        # Get key line
        key_line = metadata_lines[i].strip()
        
        # Skip empty lines or comments
        if not key_line or key_line.startswith('#'):
            i += 1
            continue
        
        # Special handling for PEOPLE and METHOD - skip them if followed by PERSONID/METHID
        if key_line in ['PEOPLE', 'METHOD', "PRODUCT"]:
            if i + 1 < len(metadata_lines):
                next_line = metadata_lines[i + 1].strip()
                if next_line in ['PERSONID', 'METHID', 'PRODID']:
                    # Skip PEOPLE/METHOD/PRODUCT, move to next line
                    i += 1
                    continue
        
        # Next line is the value
        if i + 1 < len(metadata_lines):
            value_line = metadata_lines[i + 1].strip()
            
            # Handle duplicate keys by appending _1, _2, etc. to subsequent occurrences
            if key_line in metadata_dict:
                # Key already exists, add suffix to new one
                if key_line not in key_counts:
                    key_counts[key_line] = 1
                else:
                    key_counts[key_line] += 1
                
                final_key = f"{key_line}_{key_counts[key_line]}"
            else:
                final_key = key_line
            
            # If value is empty, set to None/NaN
            if not value_line:
                metadata_dict[final_key] = None
            else:
                metadata_dict[final_key] = value_line
            
            
            # Move to next key-value pair (skip both key and value lines)
            i += 2
        else:
            # Key with no value at end of metadata
            # Handle duplicate keys here too
            if key_line in metadata_dict:
                if key_line not in key_counts:
                    key_counts[key_line] = 1
                else:
                    key_counts[key_line] += 1
                final_key = f"{key_line}_{key_counts[key_line]}"
            else:
                final_key = key_line
            
            metadata_dict[final_key] = None

            i += 1
    if metadata_dict['TABLE'] != 'CONE' and metadata_dict['TABLE'] != 'FURN': 
        raise Exception(f"This is not a Cone Calorimeter test, skipping")

    i = vector_data_start
    while i < len(lines):
        line = lines[i].strip()
        
        # Check if this is start of new variable section
        if line.startswith('VARIABLE'):
            # Next 4 lines are: raw/derived, column name, description, units
            if i + 4 < len(lines):
                raw_derived = lines[i + 1].strip()  # Ignore this
                column_name = lines[i + 2].strip()
                description = lines[i + 3].strip()  # Ignore this
                units = lines[i + 4].strip()
                
                # Create column header with units: "ColumnName (units)"
                column_header = f"{column_name} ({units})"
                
                # Collect data values until next VARIABLE or end of file
                data_values = []
                j = i + 5
                while j < len(lines):
                    data_line = lines[j].strip()
                    # Stop if we hit next VARIABLE
                    if data_line.startswith('VARIABLE'):
                        break
                    # Skip empty lines
                    if not data_line:
                        j += 1
                        continue
                    # Try to parse as float
                    try:
                        data_values.append(float(data_line))
                    except ValueError:
                        # Skip non-numeric lines
                        pass
                    j += 1
                
                # Store this column's data
                data_sections.append((column_header, data_values))
                
                # Move to where next VARIABLE starts
                i = j
            else:
                i += 1
        else:
            i += 1
    
    # Build dataframe from data sections
    if not data_sections:
        return metadata_dict, pd.DataFrame()
    
    # Create dictionary for DataFrame construction
    data_dict = {}

    # Find the Time column to use as reference length
    time_length = None
    time_header = None
    for column_header, values in data_sections:
        if 'TIME' in column_header.upper():
            time_length = len(values)
            time_header = column_header
            break

    if time_length is None:
        # Fallback to max length if no Time column found
        time_length = max(len(values) for _, values in data_sections)
        print("Warning: No Time column found, using max length as reference")

    print(f"Using reference length: {time_length} (from {time_header if time_header else 'max'})")

    for column_header, values in data_sections:
        # Special handling for COSTACK column if it's longer than time_length
        if 'COSTACK' in column_header.upper() or 'CARBON MONOXIDE' in column_header.upper():
            if len(values) == time_length + 1:
                # Check if first value is zero
                if len(values) > 0 and (values[0] == 0 or values[0] == 0.0):
                    # Remove first value
                    values = values[1:]
                    print(f"Removed leading zero from {column_header}")
        
        # Pad or truncate to match time_length
        if len(values) < time_length:
            padded_values = values + [np.nan] * (time_length - len(values))
            data_dict[column_header] = padded_values
        elif len(values) == time_length:
            data_dict[column_header] = values
        else:
            # Column is longer than time_length - truncate
            print(f"Warning: {column_header} has {len(values)} values but Time has {time_length}. Truncating.")
            data_dict[column_header] = values[:time_length]

    test_data_df = pd.DataFrame(data_dict)
        
    return test_data_df, metadata_dict
#region parse_data
def parse_data(data_df, file_name):
    """
    Convert dataframe to standard format with required columns.
    Output columns: Time (s), Mass (g), HRR (kW), MFR (kg/s), T Duct (K), O2 (Vol Fr), CO2 (Vol Fr), CO (Vol Fr), HRRPUA (kW/m2), Vduct (m3/s)
    """
    
    output_df = pd.DataFrame()
    for col in data_df.columns:
        # Time (s) - should be present as "Time"
        if 'TIME' in col.upper():
            output_df['Time (s)'] = data_df[col]

        # HEAT RELEASE RATE
        elif 'HRR/A' in col.upper() or 'HRRPUA' in col.upper():
            output_df['HRRPUA (kW/m2)'] = data_df[col]/ 1000
        elif 'HRR' in col.upper() and '/A' not in col.upper():
            output_df['HRR (kW)'] = data_df[col]/ 1000
        elif "R.H.R" in col.upper() or "RHR" in col.upper():
            output_df['HRRPUA (kW/m2)'] = data_df[col] # this one is already in kW/m2 the others are in W

        #MASS
        elif 'MASS/A' in col.upper():
            output_df['Mass LossPUA (g/m2)'] = data_df[col] * 1000 # this was original in kg/m2, convert to g/m2
        elif 'MASS' in col.upper() and '/A' not in col.upper() and "LOSS" not in col.upper() and "SOOT" not in col.upper():
            output_df['Mass (g)'] = data_df[col] * 1000 # this was original in kg, convert to g
        elif "MLR/A" in col.upper() or "MLRPUA" in col.upper():
            output_df['MLRPUA (g/s-m2)'] = data_df[col] * 1000 # this was original in kg/s-m2, convert to g/s-m2
        elif "MLR" in col.upper() and "/A" not in col.upper():
            output_df['MLR (g/s)'] = data_df[col] * 1000 # this was original in kg/s, convert to g/s
        elif "MASS LOSS RATE" in col.upper():
            output_df['MLR (g/s)'] = data_df[col] # this was original in g/s, convert to g/s
        elif "WTLOSS" in col.upper():
            output_df['MLR (g/s)'] = data_df[col] # this was original in g/s, convert to g/s 
        elif "SPECIMEN MASS" in col.upper():
            output_df['Mass (g)'] = data_df[col] # this was original in g, convert to g    


        #Flow Rates through duct (mass and volume)
        elif "VOLSTACK" in col.upper() or "VOLFLOW" in col.upper().strip():
            output_df['V Duct (m3/s)'] = data_df[col]    
        #potentially add velocity in stack, but don't know exact duct diameter so not useful
        if "FLOWDUCT" in col.upper() or "MASSFLOW" in col.upper():
            output_df['MFR (kg/s)'] = data_df[col]

        # Duct Temperature
        elif "TEMPORI" in col.upper():
            output_df['T Duct (K)'] = data_df[col]
        elif "TEMPSTCK" in col.upper() or "AVGTSTCK" in col.upper() and "T Duct (K)" not in output_df.columns:
            output_df['T Duct (K)'] = data_df[col]

        # Gas Concentrations
        elif "COYIELD" in col.upper().strip():
            output_df['CO (kg/kg)'] = data_df[col]
        elif "COSTACK" in col.upper().strip() or "CARBON MONOXIDE" in col.upper():
            output_df['CO (Vol Fr)'] = data_df[col] / 100 # convert from % to fraction
        elif "CO2YIELD" in col.upper().strip():
            output_df['CO2 (kg/kg)'] = data_df[col]
        elif "CO2STACK" in col.upper().strip():
            output_df['CO2 (Vol Fr)'] = data_df[col] / 100 # convert from % to fraction
        elif "O2STACK" in col.upper().strip() or "OXYGEN LEVEL" in col.upper():
            output_df['O2 (Vol Fr)'] = data_df[col] / 100 # convert from % to fraction
        elif "H2OSTACK" in col.upper().strip():
            output_df['H2O (Vol Fr)'] = data_df[col] / 100 # convert from % to fraction
        elif "HCLSTACK" in col.upper().strip():
            output_df['HCl (Vol Fr)'] = data_df[col] / 100 # convert from % to fraction
        elif "TUHSTACK" in col.upper().strip():
            output_df["H'carbs (Vol Fr)"] = data_df[col] / 100 # convert from % to fraction
        elif "TUH" in col.upper().strip():
            output_df["H'carbs (kg/kg)"] = data_df[col] 

        #Smoke
        elif "EXTCOEFF" in col.upper().strip("-") or "OD/M" in col.upper():
            output_df['K Smoke (1/m)'] = data_df[col]
        elif "EXAREA" in col.upper().strip() or "SEA" in col.upper():
            output_df['Extinction Area (m2/kg)'] = data_df[col]
        elif"SMOKERATE" in col.upper().strip() or "SMOKEPROD" in col.upper() or 'RSP' in col.upper():
            output_df['Smoke Production (m2/s)'] = data_df[col]
    
    # Validate time data
    times = output_df['Time (s)'].dropna().values
    if len(times) == 0:
        raise Exception("No valid time data found")
    
    start_0 = np.isclose(times[0], 0, atol=1)
    if not start_0:
        raise Exception(f"Test does not start at 0 seconds (starts at {times[0]}), please review file")
    
    increments = np.diff(times)
    expected_step = np.median(increments)
    continuous = np.all((increments >= expected_step * 0.1) & (increments <= expected_step * 5))
    if not continuous:
        raise Exception("Test does not have continuous time data, please review file")
    
    # Generate test filename from file name (T#### format)
    match = re.search(r'T(\d+)', file_name)
    if match:
        test_number = match.group(1).zfill(4)
        test_filename = f"test{test_number}"
    else:
        # Fallback to using file stem
        test_filename = Path(file_name).stem
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    print(colorize(f"Final columns: {list(output_df.columns)}", "green"))
    
    return output_df, test_filename

#region parse_metadata
def parse_metadata(metadata_dict, test_name):
    """
    Clean and output metadata as json.
    """
    meta_filename = test_name + ".json"
    meta_path = OUTPUT_DIR / meta_filename
    metadata_json = {}
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    
    # Initialize expected keys with None values
    expected_keys = [
        "Material ID", "Material Name", "Sample Mass (g)", "Residual Mass (g)",
        "Specimen Number", "Original Testname", "Testname", "Thickness (mm)",
        "Sample Description", "Specimen Prep", "Instrument", "Test Date", "Test Time",
        "Operator", "Director", "Sponsor", "Institution", "Report Name", "Original Source",
        'Preparsed', "Parsed", "Auto Prepared", "Manually Prepared", "SmURF", "Bad Data",
        "Autoprocessed", "Manually Reviewed Series", "Pass Review", "Published",
        "Heat Flux (kW/m2)", "Orientation", "C Factor", "Surface Area (m2)", "Grid",
        "Edge Frame", "Ignition Source", "Separation (mm)", "Test Start Time (s)",
        "Test End Time (s)", "MLR EOT Mass (g/m2)", "End of test criterion",
        "Heat of Combustion O2 (MJ/kg)", "OD Correction Factor", "Substrate",
        "Non-scrubbed", "Duct Diameter (m)", "O2 Delay Time (s)", "CO2 Delay Time (s)",
        "CO Delay Time (s)", "Ambient Temperature (°C)", "Barometric Pressure (Pa)",
        "Relative Humidity (%)", "X_O2 Initial", "X_CO2 Initial", 'X_CO Initial',
        't_ignition (s)', 't_ignition Outlier', 'm_ignition (g)', 'm_ignition Outlier',
        'Residue Yield (%)', 'Residue Yield Outlier', 'Heat Release Rate Outlier',
        'Average HRRPUA 60s (kW/m2)', 'Average HRRPUA 60s Outlier',
        'Average HRRPUA 180s (kW/m2)', 'Average HRRPUA 180s Outlier',
        'Average HRRPUA 300s (kW/m2)', 'Average HRRPUA 300s Outlier',
        'Steady Burning MLRPUA (g/s-m2)', 'Steady Burning MLRPUA Outlier',
        'Peak MLRPUA (g/s-m2)', 'Peak MLRPUA Outlier',
        'Steady Burning HRRPUA (kW/m2)', 'Steady Burning HRRPUA Outlier',
        'Peak HRRPUA (kW/m2)', 'Peak HRRPUA Outlier',
        'Total Heat Release (MJ/m2)', 'Total Heat Release Outlier',
        'Effective Heat Of Combustion (kJ/g)', 'Effective Heat Of Combustion Outlier',
        'Average Specific Extinction Area (m2/kg)', 'Average Specific Extinction Area Outlier',
        'Smoke Production Pre-ignition (m2/m2)', 'Smoke Production Pre-ignition Outlier',
        'Smoke Production Post-ignition (m2/m2)', 'Smoke Production Post-ignition Outlier',
        'Smoke Production Total (m2/m2)', 'Smoke Production Total Outlier',
        'Y_Soot (g/g)', 'Y_Soot Outlier', 'Y_CO2 (g/g)', 'Y_CO2 Outlier',
        'Y_CO (g/g)', 'Y_CO Outlier', 'Fire Growth Potential (m2/J)', 'Fire Growth Potential Outlier',
        'Ignition Energy (MJ/m2)', 'Ignition Energy Outlier', "t_flameout (s)", "t_flameout Outlier",
        'Comments', 'Data Corrections'
    ]
    
    for key in expected_keys:
        metadata_json.setdefault(key, None)
    
    metadata_json["Comments"] = []
    metadata_json['Data Corrections'] = []
    
    # Map TABLE format metadata to our standard fields
    field_mapping = {
        'FLUX': 'Heat Flux (kW/m2)',
        'THICK': 'Thickness (mm)',
        'THICKNESS':'Thickness (mm)',
        'AREA': 'Surface Area (m2)',
        'C': 'C Factor',
        'E': 'Heat of Combustion O2 (MJ/kg)',
        'OXYGEN': 'X_O2 Initial',
        'MASSI': 'Sample Mass (g)',
        'MASSF': 'Residual Mass (g)',
        'TIGN': 't_ignition (s)',
        'FLAMEOUT': 't_flameout (s)',
        'OPERATOR': 'Operator',
        'OFFICER': 'Director',
        'SPONSOR': 'Sponsor',
        'PRODUCT1': 'Material Name',
        'PRODNAME': 'Material Name',
        'TESTDATE': 'Test Date',
        'RHTEST': 'Relative Humidity (%)',
    }
    
    for table_key, standard_key in field_mapping.items():
        if table_key in metadata_dict and metadata_dict[table_key]:
            value = metadata_dict[table_key]
            # Try to convert to appropriate type
            try:
                # Check if it's scientific notation
                if 'E' in value or 'e' in value:
                    metadata_json[standard_key] = float(value)
                # Try integer first
                elif '.' not in value:
                    try:
                        metadata_json[standard_key] = int(value)
                    except:
                        metadata_json[standard_key] = value
                # Try float
                else:
                    metadata_json[standard_key] = float(value)
            except:
                # Keep as string if conversion fails
                metadata_json[standard_key] = value
    

    # Convert mass from kg to g if needed
    if metadata_json['Sample Mass (g)'] is not None and metadata_json['Sample Mass (g)'] < 1:
        metadata_json['Sample Mass (g)'] *= 1000
    if metadata_json['Residual Mass (g)'] is not None and metadata_json['Residual Mass (g)'] < 1:
        metadata_json['Residual Mass (g)'] *= 1000
    
    # Convert thickness from meters to mm - ALWAYS multiply by 1000
    if metadata_json['Thickness (mm)'] is not None and metadata_json['Thickness (mm)'] < 1:
        metadata_json['Thickness (mm)'] *= 1000

    #Convert Heatflux if nessesary
    if metadata_json['Heat Flux (kW/m2)'] is not None and metadata_json['Heat Flux (kW/m2)'] > 1000:
        metadata_json['Heat Flux (kW/m2)'] /= 1000
    
    # Determine Surface Area based on FRAME field
    if 'FRAME' in metadata_dict:
        if metadata_dict['FRAME'].upper() == 'Y':
            metadata_json['Surface Area (m2)'] = 0.0088  # m^2 when edge frame is used
        # Otherwise keep the AREA value that was already mapped
    
    # Convert heat of combustion from J/kg to MJ/kg if needed
    if metadata_json['Heat of Combustion O2 (MJ/kg)'] is not None and metadata_json['Heat of Combustion O2 (MJ/kg)'] > 100:
        metadata_json['Heat of Combustion O2 (MJ/kg)'] /= 1000000
    
    # Set orientation based on ORIENT field
    if 'ORIENT' in metadata_dict:
        if metadata_dict['ORIENT'] == 'H':
            metadata_json['Orientation'] = 'HORIZONTAL'
        elif metadata_dict['ORIENT'] == 'V':
            metadata_json['Orientation'] = 'VERTICAL'
    
    # Determine Ignition Source based on PILOT field
    if 'PILOT' in metadata_dict and metadata_dict['PILOT']:
        pilot = metadata_dict['PILOT'].upper()
        if pilot == 'Y':
            metadata_json['Ignition Source'] = 'Spark Igniter'
        elif pilot == 'N':
            metadata_json['Ignition Source'] = 'No Source'
    
    # Check comments for "Pilot Flame" mention (case insensitive)
    comments_text = ' '.join(metadata_json["Comments"])
    if 'pilot flame' in comments_text.lower():
        metadata_json['Ignition Source'] = 'Pilot Flame'
    
    # Set grid based on GRID field
    if 'GRID' in metadata_dict:
        if metadata_dict['GRID'] == 'Y':
            metadata_json['Grid'] = True
        elif metadata_dict['GRID'] == 'N':
            metadata_json['Grid'] = False
    
    # Set edge frame based on FRAME field
    if 'FRAME' in metadata_dict:
        if metadata_dict['FRAME'] == 'Y':
            metadata_json['Edge Frame'] = True
        elif metadata_dict['FRAME'] == 'N':
            metadata_json['Edge Frame'] = False
    
    # Determine Non-scrubbed based on ASCARITE field
    if 'ASCARITE' in metadata_dict:
        ascarite = metadata_dict['ASCARITE'].upper()
        metadata_json['Non-scrubbed'] = (ascarite == 'N')  # Y means scrubbed (false), N means non-scrubbed (true)
    
    # Set specimen number from FILE field (T####)
    if 'FILE' in metadata_dict:
        match = re.search(r'T?(\d+)', metadata_dict['FILE'])
        if match:
            metadata_json['Specimen Number'] = int(match.group(1))
    
    # Set institution from LABID
    if 'LABID' in metadata_dict:
        metadata_json['Institution'] = metadata_dict['LABID']
    
    # Set X_O2 Initial as fraction if it's a percentage
    if metadata_json['X_O2 Initial'] is not None and metadata_json['X_O2 Initial'] > 1:
        metadata_json['X_O2 Initial'] /= 100
    
    # Store all original metadata in comments
    for key, value in metadata_dict.items():
        metadata_json["Comments"].append(f"{key}: {value}")
    
    metadata_json['Original Testname'] = test_name
    metadata_json['Instrument'] = "NBS Cone Calorimeter"
    metadata_json['Preparsed'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    metadata_json["Original Source"] = "Box/Babrauskas"

    #Remove Bad Zeros
    potential_zero_keys = ['Surface Area (m2)','Thickness (mm)', "Relative Humidity (%)", "C Factor", "Heat of Combustion O2 (MJ/kg)",'X_O2 Initial']
    for key in potential_zero_keys:
        if metadata_json[key] == 0:
            metadata_json[key] = None
    
    # Write metadata file
    with open(meta_path, "w", encoding="utf-8") as f:
        f.write(json.dumps(metadata_json, indent=4)) 
    print(colorize(f"Generated {meta_path}", "blue"))
    
    return None
#region main
if __name__ == "__main__":
    # Write new log file at every run
    LOG_DIR = PROJECT_ROOT / "logs"
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    logfile = {}
    with open(LOG_FILE, "w", encoding="utf-8") as f:
        f.write(json.dumps(logfile, indent=4))
    print("✅ preparse_firedata_log.json created.")
    parse_dir(INPUT_DIR)