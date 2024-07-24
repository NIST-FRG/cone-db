import pandas as pd
from pathlib import Path
import json
from io import StringIO
import re
from dateutil import parser

INPUT_DIR = "./DATA/FAA"
OUTPUT_DIR = "./OUTPUT/FAA"

def parse_dir(input_dir):
    # read all files in the directory
    paths = Path(input_dir).glob("**/*.txt")

    # create output folder
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    for path in paths:
        try:
            parse_file(path)
        except Exception as e:
            print(f"Error parsing {path}: {e}")

def parse_file(file_path):
    # read in TXT file & split the text into metadata & actual data
    print(f"Parsing {file_path}:")

    with open(file_path, "r") as f:
        f = f.read()

    # groups of data are separated by 2 blank lines (aka. 3 newlines)
    groups = f.split("\n\n\n")

    # fyi: groups[-1] is the last element of the last, groups[-2] is the second to last, & so on
    raw_metadata = "\n".join(groups[0:-1])
    # raw_calc_data = groups[-2]
    raw_data = groups[-1]

    # parse metadata
    metadata = parse_metadata(raw_metadata)
    metadata_output_path = Path(OUTPUT_DIR) / f"{Path(file_path).stem}_metadata.json"

    with open(metadata_output_path, "w+") as f:
        json.dump(metadata, f, indent=4)

    # parse actual data
    df = parse_data(raw_data)

    # output data as a CSV file
    data_output_path = Path(OUTPUT_DIR) / f"{Path(file_path).stem}_data.csv"
    df.to_csv(data_output_path, index=False)

    print(f"\tData complete ({df.shape[0]}x{df.shape[1]})")

def parse_metadata(raw_metadata):
    # remove any leading/trailing whitespace on each line + random special characters
    raw_metadata_lines = re.split(r"\n+", raw_metadata)
    raw_metadata_lines = [x.strip() for x in raw_metadata_lines]
    
    raw_metadata = "\n".join(raw_metadata_lines)


    raw_metadata = raw_metadata.encode('ascii', errors='ignore').decode()
    raw_metadata = raw_metadata.replace("\x00", "")

    # discard any chunks without a colon (since the metadata is in key:value format)
    # also discard the first chunk (since it's just the file header)

    metadata = {}

    # helper functions to extract numbers & strings from the metadata using regular expressions
    def extract_num(key, input_str):
        match = re.search(key + r".*\s*:\s*(\d+\.?\d*)", input_str)
        if match:
            return float(match.group(1))
        else:
            return None

    # if multiline is true, the value will span multiple lines (specifically: for the sample material descrip.)
    # and ends when a blank line is encountered

    # by default, the function will stop looking when it finds a newline (\n)
    # if end is specified, it will stop looking when it finds that string instead
    def extract_string(key, input_str, end=""):
        if end == "":
            match = re.search(key + r"\s*:\s*(.+)\n", input_str)
        else:
            match = re.search(key + r"\s*:\s*(.+)\n+\s*" + end, input_str, flags=re.DOTALL)
        if match:
            return match.group(1).strip()
        else:
            return None

    # use regular expressions to extract the date & time
    # TODO: rewrite this using the extract_string helper function? (might be tricky b/c all three (date, time, operator) are in the same line)
    date = re.search(r"\d\d-\w{3}-\d\d", raw_metadata).group()
    time = re.search(r"\d{1,2}:\d\d((p|a)\.?m\.?)?", raw_metadata).group()

    # rather than messing around with regex stuff again, just use the dateutil parser
    date = parser.parse(f"{date} {time}", dayfirst=True)

    # get the operator name, also using regex 
    # (note that the entire string is always group 1 - hence .group(1) rather than .group(0))
    operator = extract_string("Operator", raw_metadata)

    # turn date into ISO8601 format
    metadata["date"] = date.isoformat()
    # remove all non-alphanumeric characters from the operator name
    metadata["operator"] = operator

    # TODO: it might be better to just search the entire metadata string rather than breaking it up into chunks
    # and searching those chunks for the relevant info

    # second chunk: test parameters
    metadata["test_end_time_s"] =  extract_num("Test Length", raw_metadata)
    # the original data is in m2 but we want cm2, so multiply by 10,000 to convert
    surface_area = extract_num("Sample Surface Area", raw_metadata)
    # if surface area is None, leave it as None, otherwise multiply by 10,000
    metadata["surface_area_cm2"] = 10_000 * surface_area if surface_area else None
    metadata["heat_flux_kW/m2"] = extract_num("Radiant Heat Flux", raw_metadata)
    metadata["sample_orientation"] = extract_string("Sample Orientation", raw_metadata)

    # third chunk: sample info
    metadata["sample_description"] = extract_string("Sample Material", raw_metadata, end="Test Notes")

    # fourth chunk: pretest comments
    pre_test_comments = extract_string("Pre-test Comments", raw_metadata, end="Post-test Comments")

    # fifth chunk: posttest comments
    # theoretically ... this could be bad if posttest comments included "Reduction Parameters" verbatim though
    post_test_comments = extract_string("Post-test Comments", raw_metadata, end="Reduction Parameters")

    metadata["comments"] = f"Pre-test:\n{pre_test_comments}\nPost-test:\n{post_test_comments}"

    # sixth chunk: reduction paramters:
    metadata["c_factor"] = extract_num("C-Factor", raw_metadata)
    metadata["e_mj/kg"] = extract_num("Conversion Factor", raw_metadata)

    # seventh chunk: various test results:
    metadata["time_to_ignition_s"] = extract_num("Time to Sustained Ignition", raw_metadata)
    metadata["initial_mass_g"] = extract_num("Entered Initial Specimen Mass", raw_metadata)
    metadata["final_mass_g"] = extract_num("Measured Final Specimen Mass", raw_metadata)

    return metadata

def parse_data(raw_data):

    # read in data as a tab-delimited CSV
    # stringIO used to read in string as a file
    df = pd.read_csv(StringIO(raw_data), sep="\t", header=[0, 1], on_bad_lines="warn")

    # there are two header rows (one for units, one for the labels themselves)
    # combine those two rows into one so it's easier to work with
    df.columns = [" ".join([h[0].strip(), f"({h[1].strip().replace("(", "").replace(")", "")})"]) for h in df.columns]
    
    # account for first column being index ("Scan")
    # TODO: fix this to use Scan as the actual index
    # this would break if negative time values are present
    temp = df.columns[1:]
    df = df.drop(df.columns[-1], axis=1)
    df.columns = temp

    df = df[["Time (secs)", "HRR (kW/m2)", "Mass (gm)" , "O2 (%)", "CO2 (%)", "CO (%)",]]

    df = df.rename(
        columns={
            "Time (secs)": "Time (s)",
            "Mass (gm)": "Mass (g)",
        }
    )

    return df

    

parse_dir(INPUT_DIR)
