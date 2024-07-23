import json
from pathlib import Path

import streamlit as st
from st_keyup import st_keyup

import pandas as pd

import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go

st.title("File explorer")

# get all metadata files
metadata_files = list(Path("data/metadata").glob("*.json"))

# Filter metadata files
st.subheader("Filter material metadata files")

all_materials = []
material_files = list(Path("../../materials").rglob("*.json"))

for mat_file in material_files:
    with open(mat_file, "r") as f:
        info = json.load(f)
        info = {k: str(v) for k, v in info.items()}
        info["Material ID"] = mat_file.stem
        all_materials.append(info)

mat_df = pd.DataFrame(all_materials).set_index("Material ID")

query = st_keyup("Search term", key="0")

if query:
    mask = mat_df.map(lambda x: query.lower() in str(x).lower()).any(axis=1)
    mat_df = mat_df[mask]

st.dataframe(mat_df)

# Search test metadata files
st.subheader("Filter test metadata files")

all_test_metadata = []
test_metadata_files = list(Path("../../OUTPUT/").rglob("*.json"))

for test_file in test_metadata_files:
    with open(test_file, "r") as f:
        metadata = json.load(f)
        metadata = {k: str(v) for k, v in metadata.items()}
        metadata["File name"] = test_file.stem
        all_test_metadata.append(metadata)

test_df = pd.DataFrame(all_test_metadata).set_index("File name")

# get only specific columns
test_df = test_df[
    [
        "date",
        "laboratory",
        "comments",
        "heat_flux_kW/m^2",
        "material_name",
        "specimen_description",
        "orientation",
        "grid",
        "manufacturer",
        "events",
    ]
]

query = st_keyup("Search term", key="1")

if query:
    mask = test_df.map(lambda x: query.lower() in str(x).lower()).any(axis=1)
    test_df = test_df[mask]

st.dataframe(test_df)

# Graph test metadata files
st.subheader("Graph multiple tests")

test_selector = st.multiselect(
    "Select tests to graph",
    options=test_df.index,
)

if len(test_selector) > 0:
    # Recursively search the OUTPUT directory for files with the names in test_selector
    test_files = []

    for test in test_selector:
        test_files.extend(list(Path("../../OUTPUT/").rglob(f"{test}.csv")))

    # Get list of each test data
    test_data = []
    for test_file in test_files:
        data = pd.read_csv(test_file)
        test_data.append(data)

    # Select column to graph
    column_to_graph = st.selectbox(
        "Select column to graph across tests", options=test_data[0].columns
    )

    data = pd.DataFrame()

    # Get only the data from the selected column for each test, and change the column name to the test name
    for i in range(len(test_data)):
        test = test_data[i]
        data = pd.concat(
            [data, test[column_to_graph].rename(test_files[i].stem)], axis=1
        )

    # Plot all the data on the same figure
    fig = px.line(data)

    # Plot the data on the same figure
    st.plotly_chart(fig)
