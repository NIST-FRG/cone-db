from pathlib import Path
from datetime import datetime
from shutil import rmtree

import pandas as pd
import json

import streamlit as st

from const import INPUT_DATA_PATH, OUTPUT_DATA_PATH

st.set_page_config(page_title="Metadata Editor", page_icon="📊", layout="wide")

st.title("Bulk Metadata Editor")

# maps the filename stem to the full path of the metadata file
metadata_path_map = {p.stem: p for p in list(INPUT_DATA_PATH.rglob("*.json"))}


# region load_metadata
# cache the metadata for faster loading (see here: https://docs.streamlit.io/get-started/fundamentals/advanced-concepts#caching)
@st.cache_data(show_spinner=True)
def load_metadata():

    # create a list of the contents of all the metadata files, as dicts
    all_metadata = []
    for metadata_path in metadata_path_map.values():
        all_metadata.append(json.load(open(metadata_path)))

    if len(all_metadata) == 0:
        st.error("No tests found.")
        return pd.DataFrame()

    # create a dataframe from the list of dicts, and sort it by date (ascending)
    df = pd.DataFrame(all_metadata, index=list(metadata_path_map.keys())).sort_values(
        by=["date"]
    )

    # create two new columns, one for deleting files and one for the material_id
    df["** DELETE FILE"] = False

    # get all the existing output files (i.e. the files that have already been processed)
    existing_output_files = [
        json.load(open(x)) for x in list(OUTPUT_DATA_PATH.rglob("**/*.json"))
    ]

    # # use the output files to find which material_ids have already been created and associate them with a filename
    # # so that they can be shown in the dataframe editor
    # all_ids = dict(
    #     [
    #         (Path(x["prev_filename"]).stem, x["material_id"])
    #         for x in existing_output_files
    #     ]
    # )

    # # if the row index is in the list of all_ids, set the material_id to the value in all_ids
    # df["material_id"] = df.index.map(lambda x: all_ids.get(x))

    return df


# region save_metadata
def save_metadata(df):
    bar = st.progress(0, "Loading metadata ...")

    bar.progress(0, "Saving metadata ...")
    files_saved = 0

    # Remove the ** DELETE FILE column
    df = df.drop(columns=["** DELETE FILE"])

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
st.sidebar.markdown("#### Save metadata")
st.sidebar.button("Save", on_click=lambda: save_metadata(df), use_container_width=True)
st.sidebar.button("Reload", on_click=st.cache_data.clear, use_container_width=True)
st.divider()

df = st.data_editor(
    df,
    use_container_width=True,
    height=650,
    # columns that are shown in the dataframe editor
    column_order=[
        "** DELETE FILE",
        "date",
        "material_id",
        "specimen_number",
        "heat_flux_kW/m2",
        "comments",
        "material_name",
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

st.divider()

st.markdown("#### Notes")
st.markdown(
    "Material ID should be in the following format: `<material_name_with_words_separated_by_underscores>:<report_name>`"
)

st.sidebar.markdown("#### Delete files")
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

    # Remove the ** DELETE FILE column
    df = df.drop(columns=["** DELETE FILE"])

    # Delete the existing output directory
    if OUTPUT_DATA_PATH.exists():
        rmtree(OUTPUT_DATA_PATH)

    files_exported = 0
    # Go through the dataframe row by row & save each file
    for index, row in df.iterrows():
        # convert the dataframe row back to a dictionary so it can be saved as a json file
        row = row.to_dict()

        # if the file has no material_id, just skip it
        if row.get("material_id") is None or row.get("material_id") in ["nan", ""]:
            continue

        # replace NaN with None to conform to official json format
        row = {k: v if not pd.isna(v) else None for k, v in row.items()}

        # find the path to the metadata file & include the old filename in the metadata
        path = metadata_path_map[str(index)]
        row["prev_filename"] = path.name

        # parse iso format datetime and just keep the date (no time)
        d = datetime.strptime(row["date"], "%Y-%m-%dT%H:%M:%S")
        year = d.strftime("%Y")

        # files are sorted into folder by year
        export_path = OUTPUT_DATA_PATH / year
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
            filename_parts.insert(2, f"r{row['specimen_number']}")

        # join all the filename parts together with a dash & add the file extension (.json)
        new_filename = "-".join(filename_parts) + ".json"

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


st.sidebar.markdown("#### Export metadata")
st.sidebar.button(
    "Export", on_click=lambda: export_metadata(df), use_container_width=True
)
