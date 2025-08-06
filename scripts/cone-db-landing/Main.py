import streamlit as st
from const import INPUT_DATA_PATH

st.set_page_config(page_title="NIST Cone Database", page_icon="🔬")

st.title("NIST Cone Database")

st.markdown(
    r"""
*The below information can also be found in the README.md file.*
## Setup
Will process and display .csv & .json files from the 'reviewed' folder. Folder structure doesn't matter, but please ensure that the CSV data file and the JSON metadata file for each test have the same name and are in the same folder.

## Usage
From the root of the respository, run the following command to start the app:

```
streamlit run scripts/cone-db-landing/Main.py
```

If that doesn't work, you can also try this:
```
python -m streamlit run scripts/cone-db-landing/Main.py
```

## Tools

#### Material IDs
Material IDs should use the format `<material_name>:<report_identifier>`. Spaces in a material name should be replaced with underscores. For example, the material ID for a material named "Material A" with a report identifier of "smith2015characterization" would be `Material_A:smith2015characterization`.

### View
Plots data for multiple tests at a time.

## General troubleshooting
If you encounter any unexpected errors, try interrupting the program (Ctrl+C) and restarting it. Note you will lose any unsaved changes in the metadata editor.
"""
)

# Make sure the INPUT_DATA_PATH and OUTPUT_DATA_PATH exist, if not create them
if not INPUT_DATA_PATH.exists():
    INPUT_DATA_PATH.mkdir(parents=True, exist_ok=True)
