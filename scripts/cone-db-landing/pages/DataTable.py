import json
from pathlib import Path

import pandas as pd

import streamlit as st
from st_keyup import st_keyup

from const import INPUT_DATA_PATH

st.set_page_config(page_title="Cone Data Table", page_icon="🗂️", layout="wide")

st.title("Cone Data Table")

# Get the paths to all the test metadata files
metadata_name_map = {p.stem: p for p in list(INPUT_DATA_PATH.rglob("*.json"))}

# Get the metadata for each test
test_metadata = []
for metadata_path in metadata_name_map.values():
    m = json.load(open(metadata_path))
    m = {k: m[k] for k in m.keys() & {"material_id", "t_ign_s", "soot_average_g/g", "peak_q_dot_kw/m2", "mf/m0_g/g", "MLRPUA"}}
    test_metadata.append(m)

# Create a dataframe with all the test metadata so that it can be easily displayed
metadata_df = pd.DataFrame(test_metadata)
metadata_df = metadata_df.rename(columns={"material_id" : "Material_ID", "peak_q_dot_kw/m2" : "HRR_peak (kW/m2)", "t_ign_s" : "t_ign (s)", "mf/m0_g/g" : "mf/m0 (g/g)", "MLRPUA" : "MLRPUA (g/s-m2)"})
metadata_df = metadata_df.set_index("Material_ID")

query = st_keyup("Search test metadata:", placeholder="e.g. 'PMMA'")

if query:
    # Filter the metadata based on the query
    mask = metadata_df.map(lambda x: query.lower() in str(x).lower()).any(axis=1)
    metadata_df = metadata_df[mask]

# Display the filtered metadata
st.dataframe(metadata_df)