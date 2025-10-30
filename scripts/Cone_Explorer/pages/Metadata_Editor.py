from pathlib import Path
from datetime import datetime
from shutil import rmtree
import os
import re
import pandas as pd
import json
import zipfile
import numpy as np
import sys
import streamlit as st
import shutil
from scipy.signal import savgol_filter
import plotly.graph_objects as go

PROJECT_ROOT = Path(__file__).resolve().parents[2]  # .../Scripts
print(PROJECT_ROOT)
sys.path.append(str(PROJECT_ROOT))
from Cone_Explorer.const import (

     PREPARED_DATA_PATH,
     PREPARED_DATA_PATH, 
     PREPARED_METADATA_PATH, 
     SCRIPT_DIR
)


################################ Title of Page #####################################################
st.set_page_config(page_title=" Cone Metadata Editor", page_icon="📊", layout="wide")
st.title("Cone Metadata Editor")


#####################################################################################################
# maps the filename stem to the full path of the metadata file
metadata_path_map = {p.stem: p for p in list(PREPARED_METADATA_PATH.rglob("*.json"))}
test_name_map = {p.stem: p for p in list(PREPARED_DATA_PATH.rglob("*.csv"))}
avg_tests = []
for test_name, test_path in test_name_map.items():
    if 'Average'  in test_name:
        avg_tests.append(test_name)
for avg in avg_tests:
    del test_name_map[avg]
############################## Get test files, select by material, then material id #################

# Get the paths to all the test files
test_material_ids = {p:p.split('_')[0] for p in test_name_map}
test_materials = {}
for matid in test_material_ids.values():
    if "-" in matid:
        test_materials[matid] = matid.split('-')[0]
    else:
        test_materials[matid] = matid

#Date Filtering
if 'start_date' not in st.session_state:
    st.session_state.start_date = "2000-01-01"  # Default date as string
if 'end_date' not in st.session_state:
    st.session_state.end_date = datetime.today().strftime("%Y-%m-%d")  # Current date as string

if st.checkbox('Filter Tests By Date'):
    st.write("Select the Date Range of Tests You Would Like to View")
    
    # Text input fields for date entries
    st.session_state.start_date = st.text_input("Start Date (YYYY-MM-DD)", value=st.session_state.start_date)
    st.session_state.end_date = st.text_input("End Date (YYYY-MM-DD)", value=st.session_state.end_date)
    
    # Validate date format
    try:
        formatted_start_date = datetime.strptime(st.session_state.start_date, "%Y-%m-%d").strftime("%Y-%m-%d")
        formatted_end_date = datetime.strptime(st.session_state.end_date, "%Y-%m-%d").strftime("%Y-%m-%d")
    except ValueError:
        st.error("Please enter valid dates in YYYY-MM-DD format.")
        st.stop()
    #Modify the test map to only include tests within the date range
    inrange_tests = []
    for test_name, test_path in metadata_path_map.items():
        with open(test_path, 'r') as f:
            metadata = json.load(f)
        if formatted_start_date <= metadata['Test Date'] <= formatted_end_date:
            inrange_tests.append(test_name)
    test_name_map = {test: test_name_map[test] for test in inrange_tests if test in test_name_map}


# Get the paths to all the metadata files
metadata_name_map = {p.stem: p for p in list(PREPARED_METADATA_PATH.rglob("*.json"))}

# Select material before selecting specific test
material_selection = st.multiselect(
    "Select a material to view:",
    options=sorted(set(test_materials.values())),   # make it a set to only keep the unique values
)

if material_selection:
    id_options = []

    for i in (material_selection):
            for t in test_materials:
                if test_materials[t] == i:
                    id_options += [t]   
    # Select material ID
    if len(id_options) == 1:
        matid_selection = st.multiselect(
            "Select versions of material to compare:",
            options=id_options, default = id_options
        )
    else:
        matid_selection = st.multiselect(
            "Select versions of material to compare:",
            options=id_options, default = None
        )

    if matid_selection:
        # Allow selection of only one test with the specidied material
        test_options = []
        for j in (matid_selection):
            for t in test_material_ids:
                if test_material_ids[t] == j:
                    test_options += [t]
        select_all = st.checkbox(f"Select All Tests for {matid_selection}", value=True)
        if select_all:
            test_selection = st.multiselect(
            "Select tests to compare:",
            options=test_options, default= test_options
                )   
        else:
            test_selection = st.multiselect(
            "Select tests to compare:",
            options=test_options,
                )

    # region load_metadata
    # cache the metadata for faster loading (see here: https://docs.streamlit.io/get-started/fundamentals/advanced-concepts#caching)
    @st.cache_data(show_spinner=False)
    ############# Function to pull in metadata and data files############################################################################
    def load_metadata():

        placeholder = st.empty()
        bar = placeholder.progress(0, "Loading metadata ...")

        # create a list of the contents of all the metadata files, as dicts
        all_metadata = []
        metadata_loaded = 0
        try:
            for i, test_stem in enumerate(test_selection):
                all_metadata.append(json.load(open(metadata_name_map[test_stem])))
                metadata_loaded += 1
                bar.progress(
                    metadata_loaded / len(test_selection),
                    f"({metadata_loaded}/{len(test_selection)}) Loading metadata for {test_stem}",
                )

            if len(all_metadata) == 0:
                st.error("Please select at least one test to view.")
                return pd.DataFrame()
            # create a dataframe from the list of dicts, and sort it by date (ascending)
            df = pd.DataFrame(all_metadata).set_index("Testname")

            # create two new columns, one for deleting files and one for the material_id (if it doesn't exist)
            df["** DELETE FILE"] = False
            df["Passed Manual Review"] = False
            df["Failed Manual Review"] = False

            bar.progress(1.0, "Metadata loaded")

            bar.progress(0, "Loading TGA data ...")

            # load in the test data
            tests_loaded = 0
            all_test_data = []
            for i, test_stem in enumerate(test_selection):
                test_path = test_name_map[test_stem]
                with open(test_path) as f:
                    all_test_data.append(pd.read_csv(f))
                    tests_loaded += 1
                    bar.progress(
                        tests_loaded / len(test_selection),
                        f"({tests_loaded}/{len(test_selection)}) Loading test data for {test_stem}",
                    )

            bar.progress(
                1.0,
                f"Loaded {len(all_test_data)} tests",
            )

            return df.sort_values(by=["Material ID"])
        except NameError:
            st.error(f"Please select a version of {material_selection} to view.")
            return pd.DataFrame()
    ##################################################################################################################################################

    ################################### Function to save metadata after columns have been added or modified ############################################
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
            old_metadata = json.load(open(path))
            new_metadata = row.dropna().to_dict() #######Drop all nan so nothing new added, no null vs NaN issues
            for key, value in new_metadata.items():
                if key in old_metadata and old_metadata[key] != value:
                    old_metadata[key] = value             
            
            ######### Update the manually prepared things
            if new_metadata["Passed Manual Review"] == True or new_metadata["Failed Manual Review"] == True: 
                if new_metadata["Passed Manual Review"] == True and  new_metadata["Failed Manual Review"] == True:
                    st.sidebar.error(f"Test {path.stem} cannot be both passed and failed in manual review, please correct or Karen will be quite upset")
                    continue
                elif new_metadata["Passed Manual Review"] == True:
                    old_metadata["Manually Reviewed Series"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    old_metadata["Pass Review"] = True
                elif new_metadata["Failed Manual Review"] == True:
                    old_metadata["Manually Reviewed Series"] =datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    old_metadata["Pass Review"] = False
            # Save the dictionary as JSON
            with open(path, "w") as f:
                json.dump(old_metadata, f, indent=4)
            files_saved += 1
            bar.progress(
                files_saved / len(metadata_path_map),
                f"Saving metadata for {path.stem}",
            )
            st.sidebar.success(f"Metadata saved to {path.stem}.")
        bar.progress(1.0, "Metadata saved")
  
    def reautoprocess(scriptname):
        '''''
        Function activated when the user clicks the "Reautoprocess" button.
        This function will go up one level to the STA scripts, run the autoprocessing script 
        which should only autoprocess changed/reviewed datasets, then jumps back into the explorer directory
        '''
        script_dir = PROJECT_ROOT / "Scripts" / "STA" 
        str_script_dir = str(script_dir.as_posix())
        script = script_dir / scriptname
        if str_script_dir not in sys.path:
            sys.path.insert(0, str_script_dir)
        with open(script, 'r') as auto:
            lines = auto.readlines()
            # Remove lines that start with 'PROJECT_ROOT ='
            filtered_lines = [line for line in lines if not line.strip().startswith("PROJECT_ROOT =")]
            source_code = "".join(filtered_lines)
        exec(source_code)

        
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

    # Reset flag and holder
    if "use_reset_df" not in st.session_state:
        st.session_state.use_reset_df = False
    if "reset_df" not in st.session_state:
        st.session_state.reset_df = None


    def reload_metadata():
        """
        Revert the editor to the original values from JSON (using cached load),
        then force the editor to remount so UI updates immediately.
        """
        st.session_state.reset_df = load_metadata()
        st.session_state.use_reset_df = True
        st.session_state.editor_key += 1
        st.rerun()


    df = refresh_meta()
    # Override df if reset was requested
    if st.session_state.use_reset_df:
        df = st.session_state.reset_df.copy()
        st.session_state.use_reset_df = False

    # --- Store original copy for reset ---
    if 'df_original' not in st.session_state:
        st.session_state.df_original = df.copy()
    st.session_state.df = df
    ###################################################################################################################

    #############################################sidebar UI/ what buttons do###############################################################

    st.sidebar.markdown("#### Select columns \nLeave blank to use defaults.")

    unlocked_columns = ["Passed Manual Review", "Failed Manual Review", "** DELETE FILE", "Comments", "Data Corrections",
                        "Institution", "Operator", "Director"]
    default_unlocked = [col for col in unlocked_columns if col in df.columns]
    selected_columns = st.sidebar.multiselect(
        "Columns",
        df.columns.tolist(),
        default=default_unlocked,
    )

    st.sidebar.markdown("#### Reload Original Metadata")
    st.sidebar.button("Reload", on_click=reload_metadata, use_container_width=True)
    st.sidebar.markdown("#### Save Metadata")
    st.sidebar.button("Save", on_click=lambda: save_metadata(df), use_container_width=True)
    st.sidebar.markdown("#### Reautoprocess Data")
    st.sidebar.button('Autoprocess', on_click=lambda: reautoprocess('Autoprocess_TGA.py'), use_container_width=True)    

    st.divider()
   
    column_config={
        "HRR (kW/m2)": st.column_config.LineChartColumn(
            "HRR (kW/m2)",
            width="medium",)
        }
  
    #lock columns that should not be edited
    for col in df.columns:
        if col not in unlocked_columns:
            column_config[col] = st.column_config.TextColumn(
                col,
                disabled=True  # Lock the column
            )
    # Data editor
    df = st.data_editor(
        df,
        key=f"data_editor_{st.session_state.editor_key}",  # forces remount
        use_container_width=True,
        column_order=selected_columns,
        column_config=column_config,
    )

    st.divider()
    st.markdown("#### Notes")
    readme = SCRIPT_DIR / "README.md"
    section_title = "### Metadata Editor"

    # Read the README file
    with open(readme, "r", encoding="utf-8") as f:
        lines = f.readlines()

    # Find start and end indices for the subsection
    start_idx, end_idx = None, None
    for i, line in enumerate(lines):
        if line.strip() == section_title:
            start_idx = i +1
            break

    if start_idx is not None:
        for j in range(start_idx + 1, len(lines)):
            if lines[j].startswith("### ") or lines[j].startswith("## "):
                end_idx = j
                break
        # If no further section, use end of file
        if end_idx is None:
            end_idx = len(lines)
        subsection = "".join(lines[start_idx:end_idx])
        st.markdown(subsection)

    st.sidebar.markdown("#### Delete files")
    ###############################################################################################################################

    ######################################## region delete_files ###############################################################
    def delete_files():
        # get all files where the ** DELETE FILE column is True
        files_to_delete = df[df["** DELETE FILE"]].index
        # delete the metadata files as well as their corresponding csv files
        for file in files_to_delete:
            metadata_path_map[file].unlink()
            test_name_map[file].with_suffix(".csv").unlink()
        # clear the cache so that the metadata files are reloaded
        st.cache_data.clear()
        st.success(f"{len(files_to_delete)} files deleted")


    st.sidebar.button("Delete files", on_click=delete_files, use_container_width=True)
    #############################################################################################################################################
