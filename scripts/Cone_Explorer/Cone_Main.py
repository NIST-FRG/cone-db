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
with open ('README.md', 'r') as f:
    st.markdown(f.read())
# Make sure the INPUT_DATA_PATH and OUTPUT_DATA_PATH exist, if not create them
if not INPUT_DATA_PATH.exists():
    INPUT_DATA_PATH.mkdir(parents=True, exist_ok=True)
