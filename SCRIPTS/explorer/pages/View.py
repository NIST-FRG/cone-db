import json

import pandas as pd

import streamlit as st

import plotly.express as px
import plotly.graph_objects as go

from const import INPUT_DATA_PATH

st.set_page_config(page_title="Cone Data Viewer", page_icon="📈")

st.title("Cone Data Viewer")

# Get the paths to all the test files
test_name_map = {p.stem: p for p in list(INPUT_DATA_PATH.rglob("*.csv"))}

# Get the paths to all the metadata files
metadata_name_map = {p.stem: p for p in list(INPUT_DATA_PATH.rglob("*.json"))}

test_selection = st.multiselect(
    "Select tests to compare:",
    options=test_name_map.keys(),
)

if len(test_selection) != 0:
    # Get the data & metadata for each test
    test_data = []
    test_metadata = []

    for test_stem in test_selection:
        test_data.append(pd.read_csv(test_name_map[test_stem]))
        test_metadata.append(json.load(open(metadata_name_map[test_stem])))

    # Select which column(s) to graph
    columns_to_graph = st.multiselect(
        "Select column(s) to graph across tests",
        options=test_data[0].columns,
        max_selections=2,
    )

    # Create plots

    for column in columns_to_graph:
        # fig = px.line(test_data[0][column].rename(test_selection[0]))
        fig = go.Figure()

        # Add additional traces for each test

        for i in range(0, len(test_data)):
            fig.add_trace(
                go.Scatter(
                    y=test_data[i][column],
                    name=test_selection[i],
                )
            ).update_layout(yaxis_title=column, xaxis_title="Time (s)")

        st.markdown(f"#### {column}")
        st.plotly_chart(fig)
