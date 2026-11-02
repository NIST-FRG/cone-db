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

INPUT_DIR = PROJECT_ROOT / "data" / "raw" / "Box" / "md_B_new"### WILL BE FIREDATA IN BOX SUBFOLDER, (firedata/flammabilitydata/cone/Box/md_B_new)
OUTPUT_DIR = PROJECT_ROOT / "data" / "preparsed" / "Box" / "md_B"
LOG_FILE = PROJECT_ROOT / "preparse_md_B_2_log.json"



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
        out_path = False
        if pct == 100:
            print(colorize(f"Parsed {path} successfully\n", "green"))
            files_parsed_fully += 1
            out_path = Path(str(path).replace('md_B', 'md_B_preparsed'))
        elif pct == 0 or pct == None:
            print(colorize(f"{path} could not be parsed", "red"))
            #out_path = Path(str(path).replace('md_B', 'md_B_bad'))
        else:
            print(colorize(f'{pct}% of tests in {path} parsed succesfully\n', 'yellow'))
            files_parsed_partial += 1
            #out_path = Path(str(path).replace('md_B', 'md_B_partial'))

        # If output path is set, ensure the directory exists and copy
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
    try: 
        tests = get_tests(lines)
        input("Press Enter to continue...")
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
        
        # Pattern: anything (###) at end of line = test number and allow for versions of same test number
        test_match = re.search(r"\((\d{3,4}[a-zA-Z]?)\)\s*$", line)
        
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
#region get_data - MODIFIED FOR NEW FORMAT

# outputting dataframe to csv file
#region get_data - MODIFIED FOR NEW FORMAT
def get_data(data):
    """
    Separates metadata from tabular data.
    Groups tables by column set, concatenates within groups, merges between groups by TIME.
    Merges duplicate columns by taking the first non-NaN value.
    Summary tables go into metadata.
    """

    # -------------------------------------------------------------------------
    # Helper: normalize a raw column name string to a standard output name
    # Raises Exception on unrecognised column
    # -------------------------------------------------------------------------
    def normalize_column(col: str, table_idx: int) -> str:
        col_upper = col.upper().strip()

        if not col_upper:
            raise Exception(f"Empty column name in table {table_idx + 1}")

        if "TIME" in col_upper:
            return "Time (s)"
        elif "DOT" in col_upper:
            return "HRRPUA (kW/m2)"
        elif "SUM Q" in col_upper or "SLM Q" in col_upper:
            return "THRPUA (MJ/m2)"
        elif "MASS" in col_upper and "LOSS" not in col_upper:
            return "Mass (g)"
        elif "LOSS" in col_upper:
            return "MLR (g/s)"
        elif "AIR" in col_upper:
            return "MFR (kg/s)"
        elif ("COM" in col_upper or "CON" in col_upper):
            return "HT Comb (MJ/kg)"
        # CO2 before CO
        elif "CO2" in col_upper or "CO₂" in col_upper or "C02" in col_upper:
            return "CO2 (kg/kg)"
        # H2O before other H patterns
        elif "H2O" in col_upper or "H₂O" in col_upper or "H20" in col_upper:
            return "H2O (kg/kg)"
        # CO after CO2 ruled out
        elif ("CO" in col_upper or "C0" in col_upper) and not any(
            x in col_upper for x in ["2", "H", "S", "M", "CARB", "CARS", "CARD"]
        ):
            return "CO (kg/kg)"
        elif any(x in col_upper for x in ["CARB", "CARS", "CARD", "CARE", 'CAR3', "CAR8"]):
            return "H'carbs (kg/kg)"
        elif "HCL" in col_upper:
            return "HCl (kg/kg)"
        elif "M-DUCT" in col_upper and "M3" not in col_upper:
            return "MFR (kg/s)"
        elif "M-DUCT" in col_upper and "M3" in col_upper:
            return "V Duct (m3/s)"
        elif "V-DUCT" in col_upper:
            return "V Duct (m3/s)"
        elif any(x in col_upper for x in ["SOOT", "SCOT", "SOCT", 'SCCT']) :
            return "Soot (kg/kg)"
        elif "AREA" in col_upper and ("SUM" not in col_upper and "SLM" not in col_upper):
            return "Extinction Area (m2/kg)"
        elif ("AREA" in col_upper and ("SUM" in col_upper or "SLM" in col_upper)) or (
            "TOTALSMOKE" in col_upper.replace(" ", "")
        ):
            return "Total Smoke (m2/kg)"
        elif "SAMPTEMP" in col_upper.replace(" ", ""):
            return "Sample Temperature (Deg C)"
        else:
            raise Exception(
                f'Unrecognised column in table {table_idx + 1}: '
                f'"{col}" (normalized: "{col_upper}")'
            )

    # -------------------------------------------------------------------------
    # Step 1: Split raw lines into metadata lines and table blocks
    # Uses the block-aware markdown parser so Format A/B/C all work correctly
    # -------------------------------------------------------------------------
    metadata = []
    table_blocks = []       # list of list[str]  (raw markdown lines per block)
    current_block = []
    in_table = False
    in_summary_table = False

    SUMMARY_KEYWORDS = {
        "MAX", "PARAMETER", "YIELD", "PEAK", "AVERAGE",
        "IGNITION", "TOTAL HEAT RELEASE", "MASS WEIGHTED", "DESCRIPTION",
        'UNIT'
    }

    for line in data:
        stripped = line.strip()
        is_pipe_line = stripped.startswith("|")

        if is_pipe_line:
            line_upper = stripped.upper()

            if "TIME" in line_upper:
                is_summary = any(kw in line_upper for kw in SUMMARY_KEYWORDS)

                if is_summary:
                    if current_block:
                        table_blocks.append(current_block)
                        current_block = []
                    in_table = False
                    in_summary_table = True
                    metadata.append(line)
                else:
                    if current_block:
                        table_blocks.append(current_block)
                    current_block = [line]
                    in_table = True
                    in_summary_table = False

            elif in_summary_table:
                metadata.append(line)

            elif in_table:
                current_block.append(line)

            else:
                metadata.append(line)

        else:
            # Non-pipe line
            if in_summary_table:
                in_summary_table = False

            if in_table:
                if not stripped:
                    # Blank line — do NOT close block yet.
                    # Look ahead: if any of the next few lines is a pipe line,
                    # keep the block open. Only close if we hit real non-pipe content.
                    current_block.append(line)   # keep blank line in block (parser will ignore it)
                else:
                    # Real non-pipe non-blank content — close block
                    table_blocks.append(current_block)
                    current_block = []
                    in_table = False
                    metadata.append(line)
            else:
                metadata.append(line)

    # Flush last open block
    if current_block:
        table_blocks.append(current_block)

    print(colorize(f"Found {len(table_blocks)} time-series table blocks", "cyan"))
    print(colorize(f"Captured {len(metadata)} metadata lines", "cyan"))

    # -------------------------------------------------------------------------
    # Step 2: Parse each table block using the format-aware cell parser
    # -------------------------------------------------------------------------
    def parse_block_to_df(block_lines: list, table_idx: int) -> pd.DataFrame:
        """
        Convert a list of raw markdown lines (one table block) to a DataFrame.

        Handles all known formats including:
        - Units row separated from header by blank line
        - Separator row appearing after first data row instead of after headers
        - Format A: units merged in header
        - Format B: units row before separator
        - Format C: units row after separator
        """

        def _is_separator(line):
            cells = [c.strip() for c in line.strip().strip('|').split('|')]
            non_empty = [c for c in cells if c]
            return bool(non_empty) and all(
                re.fullmatch(r':?-+:?', c) for c in non_empty
            )

        def _is_units_only(line):
            cells = [c.strip() for c in line.strip().strip('|').split('|')]
            non_empty = [c for c in cells if c]
            return bool(non_empty) and all(
                re.fullmatch(r'\(.*\)', c) for c in non_empty
            )

        def _is_data_row(line):
            """A data row has at least one cell that is numeric."""
            cells = [c.strip() for c in line.strip().strip('|').split('|')]
            non_empty = [c for c in cells if c]
            return any(re.fullmatch(r'[+-]?\d+\.?\d*([eE][+-]?\d+)?', c)
                    for c in non_empty)

        def _parse_row(line):
            return [c.strip() for c in line.strip().strip('|').split('|')]

        # Strip blank lines but remember which lines are pipe lines
        lines = [l for l in block_lines if l.strip()]

        if not lines:
            return pd.DataFrame()

        # -------------------------------------------------------------------------
        # Scan from the top, collecting structural rows until we hit real data.
        # Structural rows = header, units, separator (in any order, any gaps).
        # The first numeric row marks the start of data.
        # -------------------------------------------------------------------------
        header_line  = None
        units_line   = None
        data_start   = None

        for i, line in enumerate(lines):
            if not line.strip().startswith('|'):
                continue
            if _is_separator(line):
                continue  # skip separators wherever they appear
            if _is_units_only(line):
                units_line = line
                continue
            if _is_data_row(line):
                data_start = i
                break
            # Must be a header row (has text, not units-only, not separator, not data)
            header_line = line

        if header_line is None or data_start is None:
            print(colorize(
                f"Table {table_idx + 1}: could not identify header or data rows, skipping",
                "yellow"
            ))
            return pd.DataFrame()

        # -------------------------------------------------------------------------
        # Build column names
        # -------------------------------------------------------------------------
        header_cells = _parse_row(header_line)

        if units_line is not None:
            units_cells = _parse_row(units_line)
            merged = []
            for h, u in zip(header_cells, units_cells):
                if u and not h.endswith(u):
                    merged.append(f"{h} {u}".strip())
                else:
                    merged.append(h)
            header_cells = merged

        # -------------------------------------------------------------------------
        # Collect data rows — skip separators, skip non-pipe lines
        # -------------------------------------------------------------------------
        data_rows = []
        for line in lines[data_start:]:
            if not line.strip().startswith('|'):
                continue
            if _is_separator(line):
                continue
            cells = _parse_row(line)
            while len(cells) < len(header_cells):
                cells.append('')
            cells = cells[:len(header_cells)]
            data_rows.append(cells)

        if not data_rows:
            return pd.DataFrame(columns=header_cells)

        df = pd.DataFrame(data_rows, columns=header_cells)

        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        return df

    # -------------------------------------------------------------------------
    # Step 3: Parse blocks, normalise column names, handle duplicates
    # -------------------------------------------------------------------------
    dfs = []

    for idx, block in enumerate(table_blocks):
        print(colorize(f"Processing table block {idx + 1}...", "yellow"))

        raw_df = parse_block_to_df(block, idx)

        if raw_df.empty:
            print(colorize(f"Table {idx + 1}: empty after parsing, skipping", "yellow"))
            metadata.extend(block)
            continue

        # Normalize column names
        new_columns = []
        for raw_col in raw_df.columns:
            norm = normalize_column(raw_col, idx)   # raises on unknown column

            # Deduplicate within this table
            if norm in new_columns:
                counter = 2
                candidate = f"{norm}__dup{counter}"
                while candidate in new_columns:
                    counter += 1
                    candidate = f"{norm}__dup{counter}"
                print(colorize(
                    f"Warning: duplicate '{norm}' in table {idx + 1}, "
                    f"stored as '{candidate}' for merging", "yellow"
                ))
                new_columns.append(candidate)
            else:
                new_columns.append(norm)

        raw_df.columns = new_columns

        # Merge intra-table duplicate columns
        raw_df = merge_duplicate_columns(raw_df)

        # Ensure Time is float
        if "Time (s)" in raw_df.columns:
            raw_df["Time (s)"] = pd.to_numeric(raw_df["Time (s)"], errors='coerce')

        dfs.append(raw_df)
        print(colorize(
            f"Table {idx + 1}: {raw_df.shape[0]} rows, {raw_df.shape[1]} cols | "
            f"{list(raw_df.columns)}", "blue"
        ))

    if not dfs:
        raise Exception("No valid time-series tables found in data")

    # -------------------------------------------------------------------------
    # Step 4: Group tables by column set, concatenate vertically within groups
    # Use insertion-order tuple (not sorted) so column order is preserved
    # -------------------------------------------------------------------------
    column_groups: dict[tuple, list[pd.DataFrame]] = {}

    for df in dfs:
        col_key = tuple(df.columns)          # preserve order
        if col_key not in column_groups:
            column_groups[col_key] = []
        column_groups[col_key].append(df)

    print(colorize(f"Grouped into {len(column_groups)} column groups", "cyan"))

    merged_groups = []
    for group_idx, (col_key, group) in enumerate(column_groups.items()):
        print(colorize(
            f"Concatenating group {group_idx + 1} "
            f"({len(group)} table(s), cols={list(col_key)})...", "yellow"
        ))
        # Simple vertical concat — no row dropping needed because parse_block_to_df
        # never includes header/units/separator rows in data_rows
        merged_group = pd.concat(group, axis=0, ignore_index=True)
        merged_groups.append(merged_group)
        print(colorize(
            f"Group {group_idx + 1}: {merged_group.shape[0]} rows, "
            f"{merged_group.shape[1]} cols", "green"
        ))

    # -------------------------------------------------------------------------
    # Step 5: Merge groups horizontally on TIME (outer join)
    # -------------------------------------------------------------------------
    print(colorize(f"Merging {len(merged_groups)} groups horizontally by TIME...", "yellow"))

    result = merged_groups[0].copy()

    for group_idx in range(1, len(merged_groups)):
        right = merged_groups[group_idx].copy()

        # Columns in right that aren't TIME
        right_data_cols = [c for c in right.columns if c != "Time (s)"]

        # Warn on conflicts (will get __crossgroup suffix from merge)
        conflicts = set(result.columns) & set(right_data_cols)
        if conflicts:
            print(colorize(
                f"Warning: cross-group duplicate columns will be merged: {conflicts}",
                "yellow"
            ))

        result = pd.merge(
            result,
            right[["Time (s)"] + right_data_cols],
            on="Time (s)",
            how="outer",
            suffixes=('', '__crossgroup')
        )

        print(colorize(
            f"After merging group {group_idx + 1}: "
            f"{result.shape[0]} rows, {result.shape[1]} cols", "green"
        ))

    # Final deduplication pass (handles __crossgroup suffixed columns)
    result = merge_duplicate_columns(result)

    # Sort by time
    if "Time (s)" in result.columns:
        result = result.sort_values("Time (s)").reset_index(drop=True)

    print(colorize(
        f"Final: {result.shape[0]} rows, {result.shape[1]} cols | "
        f"{list(result.columns)}", "green"
    ))

    return result, metadata
    


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
    continuous_mask = (increments >= expected_step * 0.1) & (increments <= expected_step * 5)
    continuous = np.all(continuous_mask)
    continuous = np.all((increments >= expected_step *.1) & (increments <= expected_step *5))
    if not continuous:
        discontinuities = np.where(~continuous_mask)[0]
        print(f"Expected step size (median): {expected_step}")
        print(f"\nDiscontinuities found at {len(discontinuities)} location(s):")
        for idx in discontinuities:
            print(f"  Between times[{idx}] and times[{idx+1}]: "
                f"{times[idx]:.2f} -> {times[idx+1]:.2f} "
                f"(increment: {increments[idx]:.2f}, "
                f"ratio to expected: {increments[idx]/expected_step:.2f}x)")
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
    metadata_json = {}
    for key in expected_keys:
        metadata_json.setdefault(key, None)
    
    metadata_json["Comments"] = []
    metadata_json["Specimen Number"] = str(test_name).split("_")[0].split("t")[-1] # just the number
    metadata_json["Material ID"] = None
    for item in metadata:
        if metadata.index(item) == 0:
            name = item.split("(", 1)[0].strip()
            metadata_json["Material Name"] = name
        if "Heat Flux (kW/m2)" not in metadata_json:
            match = None
            if "KW/M2" in item:
                match = re.search(r'(\d+\s*KW/M2)', item)
                if match:
                    substring = match.group(1)
                else:
                    # Alternative: get all characters (digits, possibly units and spaces) just before KW/M2
                    match = re.search(r'([^\s]+(?:\s*KW/M2))', item)
                    substring = match.group(1) if match else None
                metadata_json["Heat Flux (kW/m2)"] = get_number(substring, "int")
        if "FRAME" in item or "HOLDER" in item:
            metadata_json["Edge Frame"] = True
        elif "HOR" in item:
            metadata_json["Orientation"] = "HORIZONTAL"
        elif "VERT" in item:
            metadata_json["Orientation"] = "VERTICAL"
        elif "MO" in item:
            metadata_json["Sample Mass (g)"]= get_number(item,"flt")
        elif "MF" in item:
            metadata_json["Residual Mass (g)"] = get_number(item,"flt")
        elif "TIGN" in item.replace(" ", ""):
            metadata_json["t_ignition (s)"] = get_number(item,"int")
        elif re.search(r'\s*\d+\s+(([A-Z]{3})|([A-Z]{4}))\s+\d{2}', item) is not None:
            metadata_json["Test Date"] = str(item).strip()
        if "PAGE" not in item and "---" not in item:
            metadata_json["Comments"].append(item) 
        

    metadata_json['Original Testname'] = test_name
    metadata_json['Instrument'] = "NBS Cone Calorimeter"
    metadata_json['Preparsed'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    metadata_json["Original Source"] = "Box/md_B"
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
    print("✅ preparse_md_B_2_log.json created.")
    parse_dir(INPUT_DIR)