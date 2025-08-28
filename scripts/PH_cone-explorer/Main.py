import streamlit as st
from const import INPUT_DATA_PATH, OUTPUT_DATA_PATH, PARSED_METADATA_PATH, PARSED_DATA_PATH, PREPARED_DATA_PATH
import json
import shutil
import os
from pathlib import Path
st.set_page_config(page_title="NIST Cone Data Explorer", page_icon="🔬")

st.title("NIST Cone Data Explorer")
if st.button("Import Data To Explorer"):
    INPUT_DATA_PATH.mkdir(parents=True, exist_ok=True)

    for file in PARSED_METADATA_PATH.rglob("*.json"):
        with open(file, "r", encoding="utf-8") as w:  
            metadata = json.load(w)

        smurf = metadata.get("SmURF")
        bad = metadata.get("Bad Data")

        # Target destination (flat, so just use stem)
        new_file = INPUT_DATA_PATH / file.name

        csv_file = file.with_suffix(".csv").name
        # Find the CSV in either parsed/prepared data (search recursively)
        possible_csv_paths = list(PREPARED_DATA_PATH.rglob(csv_file)) + list(PARSED_DATA_PATH.rglob(csv_file))
        src_csv_path = possible_csv_paths[0] if possible_csv_paths else None

        # If test is "bad", remove files from explorer if present
        if bad is not None:
            if new_file.exists():
                new_file.unlink()
            csv_in_explorer = new_file.with_suffix(".csv")
            if csv_in_explorer.exists():
                csv_in_explorer.unlink()
            st.sidebar.error(f"{new_file.stem} has been removed from the explorer, as it was deemed a bad test.")
            continue

        # Otherwise, check whether file already exists in explorer
        if new_file.exists():
            with open(new_file, "r", encoding="utf-8") as r:
                new_metadata = json.load(r)
            badkey = next((key for key in new_metadata if metadata.get(key) != new_metadata.get(key)), None)
            if badkey is not None and smurf is None:
                st.sidebar.error(f"Warning: please export or revert changes to {file.stem}. Difference detected in {badkey}.")
                continue

        # Copy files if not skipped above
        shutil.copy(file, new_file)
        if src_csv_path:
            shutil.copy(src_csv_path, new_file.with_suffix(".csv"))

    st.success("Data and Metadata imported successfully")
st.markdown(
    r"""
*The below information can also be found in the README.md file.*
## Setup
Place .csv & .json files from the parsing scripts (e.g. FTT.py, midas.py, etc.) in the `data` folder. Folder structure doesn't matter, but please ensure that the CSV data file and the JSON metadata file for each test have the same name and are in the same folder.

## Usage
From the root of the respository, run the following command to start the app:

```
streamlit run scripts/cone-explorer/Main.py
```

If that doesn't work, you can also try this:
```
python -m streamlit run scripts/cone-explorer/Main.py
```

## Tools

### Metadata Editor
Allows for editing metadata fields through a spreadsheet/table-like interface, as well as bulk deleting files.  

The metadata editor loads metadata from the `data` folder and upon clicking **Save**, saves changes in-place (i.e. it modifies the original JSON files in the `data` folder). If you've changed the contents of the `data` folder while the app is running, use the **Reload** button so that your changes are reflected in the editor.

Files can be deleted by checking the **\*\* DELETE FILE** checkbox next to their name and then clicking the **Delete files** button in the sidebar. 

To export the final metadata files, click the **Export metadata** button in the sidebar. This will save all the tests with a `material_id` field to the `output` folder, sorted into folders by year.

#### Material IDs
Material IDs should use the format `<material_name>:<report_identifier>`. Spaces in a material name should be replaced with underscores. For example, the material ID for a material named "Material A" with a report identifier of "smith2015characterization" would be `Material_A:smith2015characterization`.

### Search
Searches metadata files for a given text query
### View
Plots data for multiple tests at a time.

## General troubleshooting
If you encounter any unexpected errors, try interrupting the program (Ctrl+C) and restarting it. Note you will lose any unsaved changes in the metadata editor.
"""
)

# Make sure the INPUT_DATA_PATH and OUTPUT_DATA_PATH exist, if not create them
if not INPUT_DATA_PATH.exists():
    INPUT_DATA_PATH.mkdir(parents=True, exist_ok=True)
