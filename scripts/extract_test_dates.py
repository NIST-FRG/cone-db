from pathlib import Path
import json

INPUT_DIR = Path(r"../metadata/md_A/parsed")
OUTPUT_DIR_CSV = Path(r"..")

#region main
if __name__ == "__main__":
    # write new log file at every run
    OUTPUT_DATE_LOG = OUTPUT_DIR_CSV / "date_log.json"
    date_dict = {}
    for file in INPUT_DIR.iterdir():
        with open(file, "r", encoding="utf-8") as w:  
            metadata = json.load(w)
        date_dict.update({
            str(file.name) : str(metadata.get("date", "N/A")) + " || " + str(metadata.get("material_name", "N/A")) + " || " + str(metadata.get("material_id", "N/A"))
        })
    
    with open(OUTPUT_DATE_LOG, "w", encoding="utf-8") as f:
        f.write(json.dumps(date_dict, indent=4))



