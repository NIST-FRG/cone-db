import json
from pathlib import Path

import pandas as pd

import streamlit as st
from st_keyup import st_keyup

import plotly.express as px
import plotly.graph_objects as go

from const import INPUT_DATA_PATH

st.set_page_config(page_title="Cone Metadata Search", page_icon="🔎", layout="wide")

st.title("Cone Metadata Search")

# Get the paths to all the test metadata files
metadata_name_map = {p.stem: p for p in list(INPUT_DATA_PATH.rglob("*.json"))}

# Get the metadata for each test
test_metadata = []
for metadata_path in metadata_name_map.values():
    m = json.load(open(metadata_path))
    m["File name"] = metadata_path.stem
    m = {k: str(v) for k, v in m.items()}
    test_metadata.append(m)

# Create a dataframe with all the test metadata so that it can be easily displayed
metadata_df = pd.DataFrame(test_metadata).set_index("File name")


query = st_keyup("Search test metadata:", placeholder="e.g. 'PMMA'")

if query:
    # Filter the metadata based on the query
    mask = metadata_df.map(lambda x: query.lower() in str(x).lower()).any(axis=1)
    metadata_df = metadata_df[mask]

# Display the filtered metadata
st.dataframe(metadata_df)
