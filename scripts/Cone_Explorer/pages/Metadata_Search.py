import json
from pathlib import Path
import pandas as pd
import streamlit as st
from st_keyup import st_keyup
import plotly.express as px
import plotly.graph_objects as go
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]  # .../Scripts

sys.path.append(str(PROJECT_ROOT))
from Cone_Explorer.const import (
    INPUT_DATA_PATH,
    PARSED_METADATA_PATH,
    PREPARED_METADATA_PATH,
    SCRIPT_DIR
)

st.set_page_config(page_title="Cone Metadata Search", page_icon="🔎", layout="wide")

st.title("Cone Metadata Search")
st.write("Select the test status you would like to view")
test_types = ['SmURFed', "All (Parsed Versions)", "Not SmURFed"]
selected_type = st.selectbox("Choose SmURF status", test_types)
if selected_type == "Not SmURFed":
    metadata_name_map = {}

    for p in PARSED_METADATA_PATH.rglob("*.json"):
        with open(p, 'r') as f:
            metadata = json.load(f)
        # Check if 'SmURF' is absent, None, or empty string
        smurf_value = metadata.get('SmURF', None)
        if not smurf_value:  # Covers None, '', [], {}, etc.
            metadata_name_map[p.stem] = p
elif selected_type == "All (Parsed Versions)":
        # Get the paths to all the test metadata files
        metadata_name_map = {p.stem: p for p in list(PARSED_METADATA_PATH.rglob("*.json"))}
else:
    # Get the paths to all the test metadata files
    metadata_name_map = {p.stem: p for p in list(PREPARED_METADATA_PATH.rglob("*.json"))}

# Get the metadata for each test
test_metadata = []
for metadata_path in metadata_name_map.values():
    m = json.load(open(metadata_path))
    m["File name"] = metadata_path.stem
    m = {k: str(v) for k, v in m.items()}
    test_metadata.append(m)

if test_metadata != []:
    # Create a dataframe with all the test metadata so that it can be easily displayed
    metadata_df = pd.DataFrame(test_metadata).set_index("File name")


    query = st_keyup("Search test metadata:", placeholder="e.g. 'PMMA'")

    if query:
        # Filter the metadata based on the query
        mask = metadata_df.map(lambda x: query.lower() in str(x).lower()).any(axis=1)
        metadata_df = metadata_df[mask]

    # Display the filtered metadata
    st.dataframe(metadata_df)


    st.markdown("#### Notes")
    readme = SCRIPT_DIR / "README.md"
    section_title = "### Metadata Search"

    # Read the README file
    with open(readme, "r", encoding="utf-8") as f:
        lines = f.readlines()

    # Find start and end indices for the subsection
    start_idx, end_idx = None, None
    for i, line in enumerate(lines):
        if line.strip() == section_title:
            start_idx = i +1
            break

    if start_idx is not None:
        for j in range(start_idx + 1, len(lines)):
            if lines[j].startswith("### ") or lines[j].startswith("## "):
                end_idx = j
                break
        # If no further section, use end of file
        if end_idx is None:
            end_idx = len(lines)
        subsection = "".join(lines[start_idx:end_idx])
        st.markdown(subsection)
else:
    st.error(f"There are no tests with the status {selected_type} available.")