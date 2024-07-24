import json
from pathlib import Path

import streamlit as st
from st_keyup import st_keyup


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


@st.cache_data
def get_paths():
    return list(INPUT_PATH.rglob("*.csv"))


def next_file():
    st.session_state.index += 1


def prev_file():
    if st.session_state.index > 0:
        st.session_state.index -= 1


path = get_paths()[st.session_state.index]
st.title(path.stem)

# load df
df = pd.read_csv(path)

# select columns to display in chart
st.subheader("Plot data")
col_select = st.multiselect(
    "**Select columns to plot** (max 2)", df.columns, max_selections=2
)
# update session state with columns selected in col_select
st.session_state.cols = col_select

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

st.plotly_chart(fig)

# Load metadata from .json file
metadata = json.load(open(path.with_suffix(".json")))
st.subheader("Edit metadata")
st.markdown(
    "**Metadata file is updated automatically.** Leave the cell empty for `None`. `True` and `False` are converted to their boolean equivalents."
)
# Display metadata
converted_metadata = {k: str(v or "") for k, v in metadata.items()}
converted_metadata = pd.DataFrame(
    converted_metadata.items(), columns=["property", "value"]
)
converted_metadata = converted_metadata.set_index("property")
converted_metadata = st.data_editor(converted_metadata, use_container_width=True)

new_metadata = converted_metadata.to_dict()["value"]





# update metadata to have the correct types
new_metadata = dict(map(lambda kv: fix_types(kv[0], kv[1]), new_metadata.items()))

st.sidebar.header("File controls")
# st.sidebar.selectbox(
#     "Select file",
#     [x.stem for x in get_paths()],
#     index=st.session_state.index,
#     on_change=set_file,
# )
st.sidebar.button("Next file", on_click=next_file, use_container_width=True)
st.sidebar.button("Previous file", on_click=prev_file, use_container_width=True)

st.subheader("Associate with report")

st.markdown("Leave report identifier empty to not associate with a report.")
report_id = st.text_input("**Report identifier** (e.g. `emil1989flammability`)")

if not report_id.strip() == "":
    new_metadata["report_id"] = report_id
else:
    new_metadata["report_id"] = None

st.subheader("Associate with a material file")

# TODO: add material file association options:
# option 1: associate with existing material file
# option 2: associate with new material file

existing_mat, new_mat = st.tabs(
    ["Associate w/ existing material file", "Create new material file"]
)

with existing_mat:
    # Filter existing material metadata files
    existing_mat_files = list(Path("../../materials").rglob("*.json"))

    all_materials = []

    for mat_file in existing_mat_files:
        with open(mat_file, "r") as f:
            info = json.load(f)
            info = {k: str(v) for k, v in info.items()}
            info["Material ID"] = mat_file.stem
            all_materials.append(info)

    # turn into a dataframe so it can be easily filtered
    material_df = pd.DataFrame(all_materials)
    material_df.set_index("Material ID", inplace=True)

    st.markdown("##### Search for existing material file")

    query = st_keyup("Search term")

    if query:
        mask = material_df.map(lambda x: query.lower() in str(x).lower()).any(axis=1)
        material_df = material_df[mask]

    st.dataframe(material_df)

    material_id = st.selectbox(
        "Select material", [x["Material ID"] for x in all_materials]
    )

with new_mat:
    with st.form("create_mat"):
        material_id = st.text_input(
            "**Unique material identifier (used for filename & saved in metadata file)** (e.g. `HDPE`)"
        )

        common_names = st.text_input("**Common names (separate with commas)**")
        common_names = [x.strip() for x in common_names.split(",")]

        manufacturer = st.text_input("**Manufacturer**")
        distributor = st.text_input("**Distributor**")

        date_acquired = st.text_input("**Date acquired**")
        description = st.text_area("**Description**")

        # Add a way to add categories, preferably from some sort of dropdown list

        st.form_submit_button("Save")

    # check to see if material id is actually unique

    def make_new_mat_file():
        if material_id.lower() in [
            x.stem.lower() for x in list(Path("../../materials").rglob("*.json"))
        ]:
            st.error("Material ID already exists, file was not written.")
            return

        # Add material identifier to the metadata file for the test
        new_metadata["material_id"] = material_id

        material_metadata = {
            "Common Names": common_names,
            "Manufacturer": manufacturer,
            "Distributor": distributor,
            "Date Acquired": date_acquired,
            "Description": description,
        }

        # turn into a string
        json_str = json.dumps(material_metadata, indent=4)

        Path("../../materials").mkdir(parents=True, exist_ok=True)

        with open(f"../../materials/{material_id}.json", "w") as f:
            f.write(json_str)

        st.markdown(f"*Saved following JSON to `{material_id}.json`*")
        st.json(json_str)

    make_new_mat_file()

# save metadata to JSON file
with open(path.with_suffix(f".json"), "w") as f:
    # Remove key/value pairs with None values
    json.dump(new_metadata, f, indent=4)
