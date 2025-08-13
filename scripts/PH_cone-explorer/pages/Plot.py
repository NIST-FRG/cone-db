import json

import pandas as pd
import numpy as np

import streamlit as st

import plotly.express as px
import plotly.graph_objects as go

from const import INPUT_DATA_PATH

st.set_page_config(page_title="Cone Data Viewer", page_icon="📈", layout="wide")

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

    x_axis_column = st.selectbox(
        "Select the column for the x-axis",
        options=['Time (s)', 't * EHF (kJ/m2)'],
    )
    # Select the y-axis columns
    y_axis_columns = st.multiselect(
        "Select column(s) for the y-axis across tests",
        options=['HRR (kW/m2)', 'MLR (g/s-m2)', 'THR (MJ/m2)'],
    )
    # Create plots
    if x_axis_column and y_axis_columns:
        for y_column in y_axis_columns:
            fig = go.Figure()
            # Add additional traces for each test
            for i in range(len(test_data)):
                fig.add_trace(
                    go.Scatter(
                        x=test_data[i][x_axis_column],  # Use selected x-axis column
                        y=test_data[i][y_column],  # Use the current y-axis column
                        name=test_selection[i],
                    )
                )
            
            # Update layout for the current figure
            fig.update_layout(
                yaxis_title=y_column,
                xaxis_title=x_axis_column,
            )
            
            st.markdown(f"#### {y_column} vs {x_axis_column}")
            st.plotly_chart(fig)