from pathlib import Path
from datetime import datetime
from shutil import rmtree
import os

import pandas as pd
import json
import zipfile
import numpy as np

import streamlit as st
import shutil
from const import INPUT_DATA_PATH, OUTPUT_DATA_PATH, PARSED_METADATA_PATH, PREPARED_DATA_PATH, PREPARED_METADATA_PATH

################################ Title of Page #####################################################
st.set_page_config(page_title="Metadata Editor", page_icon="📊", layout="wide")
st.title("Bulk Metadata Editor")

#####################################################################################################

# maps the filename stem to the full path of the metadata file
metadata_path_map = {p.stem: p for p in list(INPUT_DATA_PATH.rglob("*.json"))}
if st.checkbox('SmURF Filter'):
    st.write("Select the test status you would like to view")
    test_types = ['SmURFed', "Not SmURFed"]
    selected_type = st.selectbox("Choose SmURF status", test_types)
    # Filter tests based on selected types
    filtered_tests = []
    if selected_type == 'SmURFed':
        for test_name, test_value in metadata_path_map.items():     
            with open(test_value, 'r') as f:
                metadata = json.load(f)
            if metadata["SmURF"] != None:
                filtered_tests.append(test_name)
    else:
        for test_name, test_value in metadata_path_map.items():     
            with open(test_value, 'r') as f:
                metadata = json.load(f)
            if metadata["SmURF"] == None:
                filtered_tests.append(test_name)
    metadata_path_map = {test: metadata_path_map[test] for test in filtered_tests if test in metadata_path_map}

if "metadata_loaded_once" not in st.session_state:
    st.session_state.metadata_loaded_once = False
# region load_metadata
# cache the metadata for faster loading (see here: https://docs.streamlit.io/get-started/fundamentals/advanced-concepts#caching)
@st.cache_data(show_spinner=False)
def load_metadata(show_bar=True):
    """Load all metadata and test data, return as a dataframe."""
    if show_bar:
        placeholder = st.empty()
        bar = placeholder.progress(0, "Loading metadata ...")
    else:
        bar = None

    all_metadata = []
    all_surf_areas = []
    metadata_loaded = 0
    for metadata_path in metadata_path_map.values():
        one_metadata = json.load(open(metadata_path))
        surf_area = one_metadata["Surface Area (m2)"]
        all_surf_areas.append(surf_area)
        all_metadata.append(one_metadata)
        metadata_loaded += 1
        if bar:
            bar.progress(
                metadata_loaded / len(metadata_path_map),
                f"({metadata_loaded}/{len(metadata_path_map)}) Loading metadata for {metadata_path.stem}",
            )

    if len(all_metadata) == 0:
        st.error("No tests found.")
        return pd.DataFrame()

    df = pd.DataFrame(all_metadata, index=list(metadata_path_map.keys()))
    df["** DELETE FILE"] = False
    df["** EXPORT FILE"] = False
    df["** REVERT FILE"] = False
    if "Material ID" not in df.columns:
        df["Material ID"] = None

    if bar:
        bar.progress(1.0, "Metadata loaded")
        bar.progress(0, "Loading HRR data ...")

    all_test_data = []
    tests_loaded = 0
    for metadata_path in metadata_path_map.values():
        test_path = metadata_path.with_suffix(".csv")
        with open(test_path) as f:
            all_test_data.append(pd.read_csv(f))
            tests_loaded += 1
            if bar:
                bar.progress(
                    tests_loaded / len(metadata_path_map),
                    f"({tests_loaded}/{len(metadata_path_map)}) Loading test data for {test_path.stem}",
                )


    for i, test_data in enumerate(all_test_data):
        if 'HRR (kW)' in test_data.columns:
            test_data['HRRPUA (kW/m2)'] = pd.to_numeric(
                test_data['HRR (kW)'], errors='coerce'
            ) / all_surf_areas[i]  # or * if that's appropriate
        else:
            test_data['HRRPUA (kW/m2)'] = pd.to_numeric(
                test_data['HRRPUA (kW/m2)'], errors='coerce'
            )
    if len(all_test_data) > 1:
        hrr = pd.concat([test_data["HRRPUA (kW/m2)"] for test_data in all_test_data], axis=1)
        hrr.columns = metadata_path_map.keys()
        hrr = hrr.apply(lambda x: x.dropna().to_list(), axis=0)
        hrr = pd.Series(hrr.squeeze())
        df.insert(0, "HRRPUA (kW/m2)", hrr.values)
    else:
        #HRR curve still not displaying properly if one test present only, come back to this
        hrr = pd.concat([test_data["HRRPUA (kW/m2)"] for test_data in all_test_data], axis=1)
        hrr.columns = metadata_path_map.keys()
        hrr = hrr.apply(lambda x: x.dropna().to_list(), axis=0)
        hrr = pd.Series(hrr.squeeze())
        df.insert(0, "HRRPUA (kW/m2)", [hrr])
    if bar:
        bar.progress(1.0, f"Loaded {len(all_test_data)} test(s)")

    return df.sort_values(by=["Test Date"])


# region save_metadata
def save_metadata(df):
    bar = st.progress(0, "Loading metadata ...")

    bar.progress(0, "Saving metadata ...")
    files_saved = 0

    # Remove the ** DELETE FILE and other things
    df = df.drop(columns=["** DELETE FILE", "HRRPUA (kW/m2)", "** EXPORT FILE", "** REVERT FILE"])

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


def refresh_meta():
    '''''
    Function activated when the user clicks the "Reload" button.
    This function will clear the cache and reload the metadata
    '''
    st.cache_data.clear()
    df = load_metadata()
    return df

    # Editor remount key
if "editor_key" not in st.session_state:
    st.session_state.editor_key = 0

# Reload flag and holder
if "use_reload_df" not in st.session_state:
    st.session_state.use_reload_df = False
if "reload_df" not in st.session_state:
    st.session_state.reload_df = None


def reload_metadata():
    """
    Revert the editor to the original values from JSON (using cached load),
    then force the editor to remount so UI updates immediately.
    """
    st.session_state.reload_df = load_metadata(show_bar = False)
    st.session_state.use_reload_df = True
    st.session_state.editor_key += 1
    #st.rerun()

df = refresh_meta()
# Override df if reload was requested
if st.session_state.use_reload_df:
    df = st.session_state.reload_df.copy()
    st.session_state.use_reload_df = False

def revert_to_parsed():
    "Pulls in the unmodified, parsed JSON file, undoing any saved and unsaved changes that were made"
    files_to_revert = df[df['** REVERT FILE']].index
    for file in files_to_revert:
        bad_data = str(metadata_path_map[file])
        original_data = bad_data.replace(str(INPUT_DATA_PATH), str(PARSED_METADATA_PATH))
        save_path = str(metadata_path_map[file])
        shutil.copy(original_data, save_path)
        st.success(f"Metadata for {file} reverted to parsed")

# region delete_files
def delete_files():
    # get all files where the ** DELETE FILE column is True
    files_to_delete = df[df["** DELETE FILE"]].index
    # delete the metadata files as well as their corresponding csv files
    for file in files_to_delete:
        #tag parsed file as being bad so it is not re-imported
        active_file = metadata_path_map[file]
        parsed_file = Path(str(active_file).replace(str(INPUT_DATA_PATH), str(PARSED_METADATA_PATH)))
        print(parsed_file)
        with open(parsed_file, "r", encoding="utf-8") as w:  
            parsed = json.load(w) 
        parsed["Bad Data"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open (parsed_file, "w", encoding="utf-8") as w:
            json.dump(parsed, w, indent = 4)
        metadata_path_map[file].unlink()
        metadata_path_map[file].with_suffix(".csv").unlink()
    # clear the cache so that the metadata files are reloaded
    st.cache_data.clear()
    st.success(f"{len(files_to_delete)} files deleted")



# --- Store original copy for reload ---
if 'df_original' not in st.session_state:
    st.session_state.df_original = df.copy()
st.session_state.df = df


# sidebar UI
st.sidebar.button("Reload Page", on_click= reload_metadata, use_container_width=True)
st.sidebar.markdown("Discards unsaved changes and reloads in data and metadata from local explorer")
st.sidebar.button("Save Metadata", on_click=lambda: save_metadata(df), use_container_width=True)
st.sidebar.markdown("Saves all changes in the active dataframe to your local explorer")
st.sidebar.button("Revert Metadata", on_click= revert_to_parsed, use_container_width=True)
st.sidebar.markdown("Reverts metadata for selected files to their parsed state, all changes will be lost")
st.sidebar.button("Delete files", on_click=delete_files, use_container_width=True)
st.sidebar.markdown("Deletes selected files from the explorer, these will not be pushed forward")
st.divider()

st.sidebar.markdown("### Select columns \nLeave blank to view all columns.")
# adjusted for format md_A

if len(metadata_path_map)> 0:
    selected_columns = st.sidebar.multiselect(
        "Columns",
        df.columns.tolist() ,
        default=[
            "** EXPORT FILE",
            "** DELETE FILE",
            '** REVERT FILE',
            "Test Date",
            "Material ID",
            "Specimen Number",
            "Heat Flux (kW/m2)",
            "Comments",
            "Material Name",
            "HRRPUA (kW/m2)",
            "Institution",
            "C Factor",
        ],
    )

    df = st.data_editor(
        df,
        key=st.session_state.editor_key,  
        use_container_width=True,
        #height=650,
        column_order=selected_columns,
        column_config={
            "HRRPUA (kW/m2)": st.column_config.LineChartColumn(
                "HRRPUA (kW/m2)",
                width="medium",
            )
        },
    )



# region export_metadata
def export_metadata(df):
    bar = st.progress(0, "Exporting metadata ...")
    export_indices = df.index[df["** EXPORT FILE"]==True].tolist()
    
    # Remove the ** DELETE FILE, ** EXPORT FILE, HRR columns
    export_df = st.session_state.reload_df.loc[export_indices]
    export = export_df.drop(columns=["** DELETE FILE", "HRRPUA (kW/m2)", "** EXPORT FILE", "** REVERT FILE"])
    # Delete the existing output directory
    if OUTPUT_DATA_PATH.exists():
        rmtree(OUTPUT_DATA_PATH)

    files_exported = 0
    # Go through the dataframe row by row & save each file
    for index, row in export.iterrows():
        # convert the dataframe row back to a dictionary so it can be saved as a json file
        row = row.to_dict()
        # if the file has no material_id, skip it
        if row.get("Material ID") is None or row.get("Material ID") in ["nan", ""] or "/" in row.get("Material ID"):
            continue

        # replace NaN with None to conform to official json format
        # row = {k: v if not pd.isna(v) else None for k, v in row.items()}
        row = {
            k: (v if not (pd.isna(v) if np.isscalar(v) else False) else None)
            for k, v in row.items()
        }

        # if the file has no heat flux, skip it:
        if (row.get("Heat Flux (kW/m2)") is None) or row.get("Heat Flux (kW/m2)") == "Not found":
            continue

        # find the path to the metadata file & include the old filename in the metadata
        path = metadata_path_map[str(index)]

        date = row["Test Date"]
        dt_obj = datetime.strptime(date, "%d %b %Y")  # Parse the string
        row['Test Date'] = dt_obj.strftime("%Y-%m-%d")  
        row['SmURF'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")


        export_path = OUTPUT_DATA_PATH / "md_A"
        if not export_path.exists():
            export_path.mkdir(parents=True, exist_ok=True)
        if not PREPARED_DATA_PATH.exists():
            PREPARED_DATA_PATH.mkdir(parents=True, exist_ok=True)
        if not PREPARED_METADATA_PATH.exists():
            PREPARED_METADATA_PATH.mkdir(parents=True, exist_ok=True)
        material_id = row.get("Material ID")

        # replace the colon in the material_id with a dash (colons are not allowed in Windows filenames)
        material_id = material_id.replace(":", "-")

        filename_parts = [
            material_id,
            str(int(row["Heat Flux (kW/m2)"])),
            "vert" if row["Orientation"] == "vertical" else "horiz",
        ]

        # if there is a specimen number, include that in the filename
        if row.get("Specimen Number") is not None and row.get("Specimen Number") != "":
            filename_parts.insert(3, f"{row['Specimen Number']}")

        # join all the filename parts together with a dash & add the file extension (.json)
        row['Testname'] = "_".join(filename_parts)
        new_filename = "_".join(filename_parts) + ".json"
        old_filename = row["Original Testname"] + '.json'
        row.pop('Number of Fields')
        neworder = ['Material ID', 'Sample Mass (g)','Specimen Number','Testname', 
                'Instrument', 'Test Date', 'Institution','Preparsed','Parsed','Auto Prepared', 'Manually Prepared', "SmURF", 'Autoprocessed', 
                    'Manually Reviewed Series','Pass Review', 'Published', "Original Testname", "Bad Data", "Heat Flux (kW/m2)", 'Orientation', 'Material Name','Conversion Factor', 'C Factor',
                   'Surface Area (m2)','Time to Ignition (s)', 'Residual Mass (g)', 'Residue Yield (g/g)','Mass Consumed', "Soot Average (g/g)",
                   'Peak Heat Release Rate (kW/m2)', 'Peak Mass Loss Rate (g/s-m2)', 'Comments', 'Data Corrections' ]
        reordered_metadata = {key: row[key] for key in neworder}
        for key in row:
            if key not in neworder:
                reordered_metadata[key] = row[key]

        # save the metadata file, including to parsed
        with open(export_path / new_filename, "w") as f:
            json.dump(reordered_metadata, f, indent=4)
        with open(PREPARED_METADATA_PATH / new_filename, "w") as f:
            json.dump(reordered_metadata, f, indent=4)
        with open(PARSED_METADATA_PATH / old_filename, "w") as f:
            json.dump(reordered_metadata, f, indent=4)
        with open(INPUT_DATA_PATH / old_filename, "w") as f:
            json.dump(reordered_metadata, f, indent=4)

        # Get the CSV data file as well and save that in the same folder as the metadata file
        data = pd.read_csv(path.with_suffix(".csv"))
        data.to_csv(export_path / new_filename.replace(".json", ".csv"), index=False)
        data.to_csv(PREPARED_DATA_PATH/ new_filename.replace(".json", ".csv"), index=False)
 
        # update progress bar & statistics
        files_exported += 1
        bar.progress(
            files_exported / len(export_indices),
            f"Exporting {path.stem} → {new_filename}",
        )
    bar.progress(1.0, f"Tests exported ({files_exported} tests)")



st.sidebar.button("Export", on_click=lambda: export_metadata(df), use_container_width=True)
st.sidebar.markdown("Selected files are renamed, and their data and metadata are exported to the prepared stage")



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





st.divider()

st.markdown("#### Notes")
st.markdown(
    """Material ID should be in the following format: `<Material name (replace spaces with underscores)>:<Report identifier>`
    Exported filenames are in the following format: `<Material ID>-<Heat flux (kW/m2)>-r<Specimen number (if available)>-<Orientation (vert or horiz)>.json`
    These four parameters, plus the year **must** be unique for each test in order for it to be exported correctly. i.e. if two tests have the exact same material ID, heat flux, specimen number (or both have no specimen number at all), orientation and year, the 2nd test will **overwrite** the first one.
    *Note that colons are not allowed in Windows filenames, so colons in the material ID will be replaced with dashes.*
    """
)
