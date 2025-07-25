from pathlib import Path
from datetime import datetime
from shutil import rmtree
import os

import pandas as pd
import json
import zipfile
import numpy as np

import streamlit as st

from const import INPUT_DATA_PATH, OUTPUT_DATA_PATH

st.set_page_config(page_title="Metadata Editor", page_icon="📊", layout="wide")

st.title("Bulk Metadata Editor")

# maps the filename stem to the full path of the metadata file
metadata_path_map = {p.stem: p for p in list(INPUT_DATA_PATH.rglob("*.json"))}


# region load_metadata
# cache the metadata for faster loading (see here: https://docs.streamlit.io/get-started/fundamentals/advanced-concepts#caching)
@st.cache_data(show_spinner=False)
def load_metadata():

    placeholder = st.empty()
    bar = placeholder.progress(0, "Loading metadata ...")

    # create a list of the contents of all the metadata files, as dicts
    all_metadata = []
    metadata_loaded = 0
    for metadata_path in metadata_path_map.values():
        all_metadata.append(json.load(open(metadata_path)))
        metadata_loaded += 1
        bar.progress(
            metadata_loaded / len(metadata_path_map),
            f"({metadata_loaded}/{len(metadata_path_map)}) Loading metadata for {metadata_path.stem}",
        )

    if len(all_metadata) == 0:
        st.error("No tests found.")
        return pd.DataFrame()

    # create a dataframe from the list of dicts, and sort it by date (ascending)
    df = pd.DataFrame(all_metadata, index=list(metadata_path_map.keys()))

    # create two new columns, one for deleting files and one for the material_id (if it doesn't exist)
    df["** DELETE FILE"] = False

    if "material_id" not in df.columns:
        df["material_id"] = None

    bar.progress(1.0, "Metadata loaded")

    bar.progress(0, "Loading HRR data ...")

    # load in the test data
    tests_loaded = 0
    all_test_data = []
    for metadata_path in metadata_path_map.values():
        test_path = metadata_path.with_suffix(".csv")
        with open(test_path) as f:
            all_test_data.append(pd.read_csv(f))
            tests_loaded += 1
            bar.progress(
                tests_loaded / len(metadata_path_map),
                f"({tests_loaded}/{len(metadata_path_map)}) Loading test data for {test_path.stem}",
            )

    # each row in the dataframe is a test, and the "HRR (kW/m2)" column should contain all the HRR values for that test, as a pd series
    hrr = pd.concat([test_data["HRR (kW/m2)"] for test_data in all_test_data], axis=1)
    hrr.columns = metadata_path_map.keys()
    hrr = hrr.apply(lambda x: x.dropna().to_list(), axis=0)
    hrr = pd.Series(hrr.squeeze())
    df.insert(0, "HRR (kW/m2)", hrr.values)

    bar.progress(
        1.0,
        f"Loaded {len(all_test_data)} test(s)",
    )

    return df.sort_values(by=["date"])


# region save_metadata
def save_metadata(df):
    bar = st.progress(0, "Loading metadata ...")

    bar.progress(0, "Saving metadata ...")
    files_saved = 0

    # Remove the ** DELETE FILE column
    df = df.drop(columns=["** DELETE FILE", "HRR (kW/m2)"])

    # Go through the dataframe row by row & save each file
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


df = load_metadata()

# sidebar UI

st.sidebar.markdown("### Save metadata")
st.sidebar.button("Save", on_click=lambda: save_metadata(df), use_container_width=True)
st.sidebar.button("Reload", on_click=st.cache_data.clear, use_container_width=True)
st.divider()

st.sidebar.markdown("### Delete files")
st.sidebar.markdown(
    "Select files by clicking the checkbox next to the file name, then click **Delete files** to delete the selected files."
)


# region delete_files
def delete_files():
    # get all files where the ** DELETE FILE column is True
    files_to_delete = df[df["** DELETE FILE"]].index
    # delete the metadata files as well as their corresponding csv files
    for file in files_to_delete:
        metadata_path_map[file].unlink()
        metadata_path_map[file].with_suffix(".csv").unlink()
    # clear the cache so that the metadata files are reloaded
    st.cache_data.clear()
    st.success(f"{len(files_to_delete)} files deleted")


st.sidebar.button("Delete files", on_click=delete_files, use_container_width=True)


# region export_metadata
def export_metadata(df):
    bar = st.progress(0, "Exporting metadata ...")

    # Remove the ** DELETE FILE, HRR columns
    df = df.drop(columns=["** DELETE FILE", "HRR (kW/m2)"])

    # Delete the existing output directory
    if OUTPUT_DATA_PATH.exists():
        rmtree(OUTPUT_DATA_PATH)

    files_exported = 0
    # Go through the dataframe row by row & save each file
    for index, row in df.iterrows():
        # convert the dataframe row back to a dictionary so it can be saved as a json file
        row = row.to_dict()

        # if the file has no material_id, skip it
        if row.get("material_id") is None or row.get("material_id") in ["nan", ""] or "/" in row.get("material_id"):
            continue

        # replace NaN with None to conform to official json format
        # row = {k: v if not pd.isna(v) else None for k, v in row.items()}
        row = {
            k: (v if not (pd.isna(v) if np.isscalar(v) else False) else None)
            for k, v in row.items()
        }

        # if the file has no heat flux, skip it:
        if (row.get("heat_flux_kW/m2") is None) or row.get("heat_flux_kW/m2") == "Not found":
            continue

        # find the path to the metadata file & include the old filename in the metadata
        path = metadata_path_map[str(index)]
        row["prev_filename"] = path.name

        
        # parse iso format datetime and just keep the date (no time)
        #d = datetime.strptime(row["date"], "%Y-%m-%dT%H:%M:%S")
        #year = d.strftime("%Y")

        # files are sorted into folder by year
        # export_path = OUTPUT_DATA_PATH / year
        export_path = OUTPUT_DATA_PATH / "md_A"
        if not export_path.exists():
            export_path.mkdir(parents=True, exist_ok=True)

        material_id = row.get("material_id")

        # replace the colon in the material_id with a dash (colons are not allowed in Windows filenames)
        material_id = material_id.replace(":", "-")

        filename_parts = [
            material_id,
            str(int(row["heat_flux_kW/m2"])),
            "vert" if row["orientation"] == "vertical" else "horiz",
        ]

        # if there is a specimen number, include that in the filename
        if row.get("specimen_number") is not None and row.get("specimen_number") != "":
            filename_parts.insert(2, f"{row['specimen_number']}")

        # join all the filename parts together with a dash & add the file extension (.json)
        new_filename = "_".join(filename_parts) + ".json"

        # save the metadata file
        with open(export_path / new_filename, "w") as f:
            json.dump(row, f, indent=4)

        # Get the CSV data file as well and save that in the same folder as the metadata file
        data = pd.read_csv(path.with_suffix(".csv"))
        data.to_csv(export_path / new_filename.replace(".json", ".csv"), index=False)

        # update progress bar & statistics
        files_exported += 1
        bar.progress(
            files_exported / len(metadata_path_map),
            f"Exporting {path.stem} → {new_filename}",
        )
    bar.progress(1.0, f"Tests exported ({files_exported} tests)")


st.sidebar.markdown("### Export test data & metadata")
st.sidebar.button(
    "Export", on_click=lambda: export_metadata(df), use_container_width=True
)

st.sidebar.markdown("### Select columns \nLeave blank to view all columns.")


_ = """selected_columns = 
st.sidebar.multiselect(
    "Columns",
    df.columns.tolist() + ["** DELETE FILE", "material_id", "HRR (kW/m2)"],
    default=[
        "** DELETE FILE",
        "date",
        "material_id",
        "specimen_number",
        "heat_flux_kW/m2",
        "comments",
        "material_name",
        "HRR (kW/m2)",
        "specimen_description",
        "specimen_prep",
        "report_name",
        "laboratory",
        "operator",
        "test_start_time_s",
        "test_end_time_s",
        "c_factor",
    ],
)
"""


# adjusted for format md_A
selected_columns = st.sidebar.multiselect(
    "Columns",
    df.columns.tolist() + ["** DELETE FILE", "material_id", "HRR (kW/m2)"],
    default=[
        "** DELETE FILE",
        "date",
        "material_id",
        "specimen_number",
        "heat_flux_kW/m2",
        "comments",
        "material_name",
        "HRR (kW/m2)",
        "laboratory",
        "c_factor",
    ],
)


df = st.data_editor(
    df,
    use_container_width=True,
    height=650,
    # columns that are shown in the dataframe editor
    column_order=selected_columns,
    column_config={
        "HRR (kW/m2)": st.column_config.LineChartColumn(
            "HRR (kW/m2)",
            width="medium",
        )
    },
)

st.divider()

st.markdown("#### Notes")
st.markdown(
    """Material ID should be in the following format: `<Material name (replace spaces with underscores)>:<Report identifier>`
    Exported filenames are in the following format: `<Material ID>-<Heat flux (kW/m2)>-r<Specimen number (if available)>-<Orientation (vert or horiz)>.json`
    These four parameters, plus the year **must** be unique for each test in order for it to be exported correctly. i.e. if two tests have the exact same material ID, heat flux, specimen number (or both have no specimen number at all), orientation and year, the 2nd test will **overwrite** the first one.
    *Note that colons are not allowed in Windows filenames, so colons in the material ID will be replaced with dashes.*
    """
)
