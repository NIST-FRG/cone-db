from pathlib import Path
from utils import colorize
import re
import json
import shutil

INPUT_DIR = Path(r"../data/raw/Legacy")
OUTPUT_DIR_A = Path(r"../data/raw/format_A")
OUTPUT_DIR_B = Path(r"../data/raw/format_B")
LOG_FILE = Path(r"..") / "detect_log.json"

#region detect_dir
def detect_dir(input_dir):
    paths = Path(input_dir).glob("**/*.md")
    paths = list(paths)
    #paths = list(filter(lambda x: x.stem.endswith("md"), list(paths)))
    print(paths)
    total_files = len(paths)
    print(colorize(f"Found {len(paths)} files to detect", "purple"))
    files_detect = 0
    files_detect_successfully = 0
    
    # track and print detecting success rate
    for path in paths:
        if files_detect % 20 == 0 and files_detect != 0:
            print(colorize(f"Files detect successfully: {files_detect_successfully}/{files_detect} ({(files_detect_successfully/files_detect) * 100}%)", "blue"))

        try:
            files_detect += 1
            detect_file(path)
        except Exception as e:

            # log error in md_A_log
            with open(LOG_FILE, "r", encoding="utf-8") as w:  
                logfile = json.load(w)
            logfile.update({
                    str(path.name)[0:8:1] : str(e)
                })
            with open(LOG_FILE, "w", encoding="utf-8") as f:
	            f.write(json.dumps(logfile, indent=4))

            print(colorize(f" - Error detecting {path}: {e}\n", "red"))
            continue
        print(colorize(f"Detect {path} successfully\n", "green"))
        files_detect_successfully += 1


#def split_md_df(file_path): maybe move split to parts here
#region detect file   
def detect_file(file_path):
    index = 0
    dataStart = -1
    test_starts = []
    test_sections = {}

    #generating contents for log
    with open(LOG_FILE, "r", encoding="utf-8") as w:  
        logfile = json.load(w)
    
    print(colorize(f"Detecting {file_path.name}:", "yellow"))
    
    # read lines in file
    with open(file_path, "r", encoding="utf-8") as file:
        OUTPUT_DIR = OUTPUT_DIR_A
        lines = file.readlines()
        print(f"Read {len(lines)} lines from file")
        # print(lines)

    text = ''.join(lines)
    # Case-insensitive count of "max" as a whole word
    max_matches = re.findall(r"\bmax\b", text, flags=re.IGNORECASE)
    # Case for Format B
    if len(max_matches) > 4:
        OUTPUT_DIR = OUTPUT_DIR_B

    # update md_A_log based off uniformity of columns
    logfile.update({
            str(file_path.name) : str(OUTPUT_DIR)
        })

    # copy file to detected format directory    
    shutil.copy(file_path, OUTPUT_DIR / file_path.name)

    with open(LOG_FILE, "w", encoding="utf-8") as f:
	    f.write(json.dumps(logfile, indent=4))

    print(colorize(f"{file_path} detected successfully and copied to {OUTPUT_DIR / file_path.name}\n", "green"))


def reset_dir(dir_path):
    if dir_path.exists() and dir_path.is_dir():
        shutil.rmtree(dir_path)  # Deletes the directory and all contents
        #print(f"Deleted existing directory: {dir_path}")

    dir_path.mkdir(parents=True, exist_ok=True)
    print(f"Created new directory: {dir_path}")



#region main
if __name__ == "__main__":
    reset_dir(OUTPUT_DIR_A)
    reset_dir(OUTPUT_DIR_B)

    # write new log file at every run
    logfile = {}
    with open(LOG_FILE, "w", encoding="utf-8") as f:
        f.write(json.dumps(logfile, indent=4))
    print("✅ detect_log.json created.")
    detect_dir(INPUT_DIR)