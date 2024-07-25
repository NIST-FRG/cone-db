from pathlib import Path

import pandas as pd
import json

import streamlit as st

from const import INPUT_DATA_PATH

st.set_page_config(page_title="Metadata Editor", page_icon="📊", layout="wide")

st.title("Bulk Metadata Editor")

metadata_path_map = {p.stem: p for p in list(INPUT_DATA_PATH.rglob("*.json"))}

col1, col2, col3 = st.columns([0.2, 0.1, 0.7], vertical_alignment="top", gap="small")


@st.cache_data(show_spinner=True)
def load_metadata():

    all_metadata = []
    for metadata_path in metadata_path_map.values():
        all_metadata.append(json.load(open(metadata_path)))

    return pd.DataFrame(all_metadata, index=list(metadata_path_map.keys())).sort_values(
        by=["date"]
    )


def save_metadata():
    bar = col3.progress(0, "Loading metadata ...")

    bar.progress(0, "Saving metadata ...")
    # Go through the dataframe row by row & save each file
    files_saved = 0
    for index, row in df.iterrows():
        # Get the path to the metadata file
        path = metadata_path_map[str(index)]
        # Convert the dataframe row to a dictionary
        row = row.to_dict()
        # Save the dictionary as JSON
        with open(path, "w") as f:
            json.dump(row, f, indent=4)
        files_saved += 1
        bar.progress(
            files_saved / len(metadata_path_map),
            f"Saving metadata for {path.stem}",
        )
    bar.progress(1.0, "Metadata saved")


col1.button("Save", on_click=save_metadata, use_container_width=True)
col2.button("Reload", on_click=st.cache_data.clear, use_container_width=True)
st.divider()

df = load_metadata()

df = st.data_editor(df, use_container_width=True, height=650)
