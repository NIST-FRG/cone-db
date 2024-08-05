import pandas as pd
import json
from pathlib import Path
from dateutil import parser

INPUT_FOLDER = "./process/data/"

# Get all JSON (metadata) files from the folder
meta_files = list(Path(INPUT_FOLDER).rglob("*.json"))

rows = []

for path in meta_files:
    print(f"Now parsing: {path.stem}")

    with open(path, "r") as f:
        data = json.load(f)

        # Process the key names
        data = {k.replace("_", " "): v for (k, v) in data.items()}

        data["source file"] = str(path.stem)

        data["date"] = parser.parse(data["date"]).strftime("%Y-%m-%d %H:%M:%S")

        rows.append(data)

# Save the data to an Excel file
df = pd.DataFrame.from_dict(rows, orient="columns")

# Sort rows by date
df.sort_values(by=["date"], inplace=True)

# Fix the index column (column A)
df.reset_index(drop=True, inplace=True)

# Select columns

df = df[
    [
        "comments",
        "material name",
        "specimen description",
    ]
].drop_duplicates()

print("Writing to Excel file...")

df.to_excel("all_tests.xlsx")
