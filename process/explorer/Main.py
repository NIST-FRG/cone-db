import streamlit as st
from const import INPUT_DATA_PATH, OUTPUT_DATA_PATH

st.set_page_config(page_title="NIST Cone Data Explorer", page_icon="🔬")

st.title("NIST Cone Data Explorer")

st.markdown(
    """
*The below information can also be found in the README.md file.*
## Setup
Place .csv & .json files from the parsing scripts (e.g. FTT.py, midas.py, etc.) in the `data` folder. Folder structure doesn't matter, but please ensure that the CSV data file and the JSON metadata file for each test have the same name and are in the same folder.

## Usage
`cd` into this directory (from the root of the repository: `cd process`).

Run the following command to start the app:
```
streamlit run explorer/Main.py
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
# if not OUTPUT_DATA_PATH.exists():
#     OUTPUT_DATA_PATH.mkdir(parents=True, exist_ok=True)
