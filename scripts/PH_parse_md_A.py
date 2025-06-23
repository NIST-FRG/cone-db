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
            print(colorize(f" - Error parsing {path}: {e}", "red"))
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

    tests = get_tests(lines)

    

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
    


#region main
if __name__ == "__main__":
    parse_dir(INPUT_DIR)