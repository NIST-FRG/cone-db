import json
import streamlit as st
from pathlib import Path

import pandas as pd

import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go


pd.options.plotting.backend = "plotly"

INPUT_PATH = Path(r"../../OUTPUT/")

# Initialize session state
if "cols" not in st.session_state:
    st.session_state.cols = []

if "index" not in st.session_state:
    st.session_state.index = 0


@st.cache_resource
def get_paths():
    return list(INPUT_PATH.rglob("*.csv"))


bar = st.progress(
    (st.session_state.index) / len(get_paths()),
    text=f"{st.session_state.index}/{len(get_paths())} files",
)


path = get_paths()[st.session_state.index]

formatted_test_name = path.stem
st.title(formatted_test_name)

# load df
df = pd.read_csv(path)

# select columns to display in chart
st.sidebar.header("Data controls")
col_select = st.sidebar.multiselect("Select Columns", df.columns, max_selections=2)
# update session state with columns selected in col_select
st.session_state.cols = col_select

# Load metadata from .json file
metadata = json.load(open(path.with_suffix(".json")))
# remove empty key/value pairs from metadata
metadata = {k.replace("_", " "): v for k, v in metadata.items() if v}
metadata = pd.DataFrame(metadata, index=[0])


# select a few key/value pairs from metadata
st.sidebar.header("Metadata")
st.sidebar.table(metadata.T)


x_data = df["Time (s)"]

# Create plotly figure

fig = make_subplots(specs=[[{"secondary_y": True}]])
if len(st.session_state.cols) >= 1:
    fig.add_trace(
        go.Scatter(
            x=x_data, y=df[st.session_state.cols[0]], name=st.session_state.cols[0]
        ),
        secondary_y=False,
    )
if len(st.session_state.cols) == 2:
    fig.add_trace(
        go.Scatter(
            x=x_data,
            y=df[st.session_state.cols[1]],
            name=f"{st.session_state.cols[1]} (sec.)",
        ),
        secondary_y=True,
    )

# If the data looks good, save a copy into one folder

st.header("Good data?")

col1, col2 = st.columns(2)

col1.button("Yes", on_click=lambda: next_file(True), use_container_width=True)
col2.button("No", on_click=lambda: next_file(False), use_container_width=True)


def next_file(data_type):
    if data_type:
        df.to_csv(f"./data/good/{path.stem}.csv")
    else:
        df.to_csv(f"./data/bad/{path.stem}.csv")

    st.session_state.index += 1


st.plotly_chart(fig)
# st.write(df)
