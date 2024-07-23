import json

import pandas as pd

import streamlit as st

import plotly.express as px
import plotly.graph_objects as go

from const import INPUT_DATA_PATH

st.title("Cone Data Processor")

# Get the paths to all the test files
all_test_paths = list(INPUT_DATA_PATH.rglob("*.csv"))
test_name_map = {p.stem: p for p in all_test_paths}

# Get the paths to all the metadata files
all_metadata_paths = list(INPUT_DATA_PATH.rglob("*.json"))
metadata_name_map = {p.stem: p for p in all_metadata_paths}


# Initialize some session state variables
if "columns" not in st.session_state:
    st.session_state.columns = []
if "index" not in st.session_state:
    st.session_state.index = 0

st.subheader(list(test_name_map.keys())[st.session_state.index])

# Progress bar
progress_bar = st.progress(
    (st.session_state.index + 1) / len(test_name_map),
    text=f"Test {st.session_state.index + 1} of {len(test_name_map)}",
)


# File controls
def next_file():
    if st.session_state.index + 1 >= len(test_name_map):
        return
    st.session_state.index += 1


def prev_file():
    if st.session_state.index - 1 < 0:
        return
    st.session_state.index -= 1


def set_file():
    st.session_state.index = all_test_paths.index(test_name_map[file_selection])


col1, col2 = st.columns(2)
col1.button("Previous file", use_container_width=True, on_click=prev_file)
col2.button("Next file", use_container_width=True, on_click=next_file)

col3, col4 = st.columns([0.8, 0.2], vertical_alignment="bottom")
file_selection = col3.selectbox(
    "Select test",
    list(test_name_map.keys()),
    index=st.session_state.index,
)
col4.button("Go", on_click=set_file, use_container_width=True)

test_data = pd.read_csv(
    test_name_map[list(test_name_map.keys())[st.session_state.index]]
)

st.divider()

st.markdown("#### Graph data")
columns_to_graph = st.multiselect(
    "Select column(s) from test to graph",
    options=test_data.columns,
    max_selections=2,
)
