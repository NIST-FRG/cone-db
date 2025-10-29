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
sys.path.append(str(PROJECT_ROOT))
from Cone_Explorer.const import (
    INPUT_DATA_PATH,
     OUTPUT_DATA_PATH, 
     PARSED_DATA_PATH,
     PREPARED_DATA_PATH,
     PARSED_METADATA_PATH, 
     PREPARED_DATA_PATH, 
     PREPARED_METADATA_PATH, 
     SCRIPT_DIR
)
################################ Title of Page #####################################################
st.set_page_config(page_title="SmURF Editor", page_icon="📊", layout="wide")
st.title("SmURF Editor")

#####################################################################################################
if st.button("Import Data To Explorer"):
    INPUT_DATA_PATH.mkdir(parents=True, exist_ok=True)

    for file in PARSED_METADATA_PATH.rglob("*.json"):
        with open(file, "r", encoding="utf-8") as w:  
            metadata = json.load(w)

        smurf = metadata.get("SmURF")
        bad = metadata.get("Bad Data")

        # Target destination (flat, so just use stem)
        new_file = INPUT_DATA_PATH / file.name

        csv_file = file.with_suffix(".csv").name
        # Find the CSV in either parsed/prepared data (search recursively)
        possible_csv_paths = list(PREPARED_DATA_PATH.rglob(csv_file)) + list(PARSED_DATA_PATH.rglob(csv_file))
        src_csv_path = possible_csv_paths[0] if possible_csv_paths else None

        # If test is "bad", remove files from explorer if present
        if bad is not None:
            if new_file.exists():
                new_file.unlink()
            csv_in_explorer = new_file.with_suffix(".csv")
            if csv_in_explorer.exists():
                csv_in_explorer.unlink()
            st.sidebar.error(f"{new_file.stem} has been removed from the explorer, as it was deemed a bad test.")
            continue

        # Otherwise, check whether file already exists in explorer
        if new_file.exists():
            with open(new_file, "r", encoding="utf-8") as r:
                new_metadata = json.load(r)
            badkey = next((key for key in new_metadata if metadata.get(key) != new_metadata.get(key)), None)
            if badkey is not None and smurf is None:
                st.sidebar.error(f"Warning: please export or revert changes to {file.stem}. Difference detected in {badkey}.")
                continue

        # Copy files if not skipped above
        shutil.copy(file, new_file)
        if src_csv_path:
            shutil.copy(src_csv_path, new_file.with_suffix(".csv"))

    st.success("Data and Metadata imported successfully")
# maps the filename stem to the full path of the metadata file
metadata_path_map = {p.stem: p for p in list(INPUT_DATA_PATH.rglob("*.json"))}
data_path_map = {p.stem: p for p in list(INPUT_DATA_PATH.rglob("*.csv"))}
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
    data_path_map = {test: metadata_path_map[test].with_suffix('.csv') for test in filtered_tests}

if "metadata_loaded_once" not in st.session_state:
    st.session_state.metadata_loaded_once = False
if len(metadata_path_map) == 0:
    st.warning("No metadata files found.")
    st.stop()

# Let them pick a single test to view/edit:
selected_test = st.selectbox(
    "Select a test to view/edit",
    options=list(metadata_path_map.keys()),
)

# Only show this test in the app, and send just this to downstream logic:
test_selection = metadata_path_map[selected_test]
data_selection = data_path_map[selected_test]


#region data editor
st.divider()
st.subheader("Data Modification")
date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
if test_selection:
    test_metadata = json.load(open(test_selection))
    ######## Revert data in case it was clipped and you want the full dataframe from parsed back#####################
    if st.button("Revert Data Modifications"):
        ogform = test_metadata["Original Source"]
        explorer_data = str(data_selection)
        parsed_path = str(PARSED_DATA_PATH) + f"\\{ogform}"
        original_data = explorer_data.replace(str(INPUT_DATA_PATH), parsed_path)
        save_path = str(data_selection)
        shutil.copy(original_data, save_path)
        test_metadata["Data Corrections"].append(f"{date}: Data reverted to original")
        with open(test_selection, "w") as f:
            json.dump(test_metadata, f, indent=4)
        st.sidebar.success(f"Data in {save_path} reverted to original.")
        
    data = pd.read_csv(data_selection)
    surf_area = test_metadata.get("Surface Area (m2)")
    mass = test_metadata.get("Sample Mass (g)")
    data['dt'] = data["Time (s)"].diff()
    # Normal and area adjusted HRR and THR generation
    if "HRRPUA (kW/m2)" not in data.columns:
        data["HRRPUA (kW/m2)"] = data["HRR (kW)"] / surf_area if surf_area is not None else None
    data["QPUA (MJ/m2)"] = (data['HRRPUA (kW/m2)'] * data['dt']) / 1000
    data["THRPUA (MJ/m2)"] = data["QPUA (MJ/m2)"].cumsum()
    data['Q (MJ)'] = (data['HRR (kW)'] * data['dt']) / 1000
    data['THR (MJ)'] = data["Q (MJ)"].cumsum()

    # Mass and Mass Loss Rate Data
    if "MLR (g/s)" in data.columns:
        data["MLRPUA (g/s-m2)"] = data["MLR (g/s)"] / surf_area if surf_area is not None else None
        data["MassPUA (g/m2)"] = None
    elif "MLRPUA (g/s-m2)" in data.columns:
        data["MLR (g/s)"] = data["MLRPUA (g/s-m2)"] / surf_area if surf_area is not None else None
        data["MassPUA (g/m2)"] = None
    elif not data["Mass (g)"].isnull().all():
        data['MLR (g/s)'] = abs(np.gradient(data["Mass (g)"])) # COME BACK TO THIS
        data["MLRPUA (g/s-m2)"] = data["MLR (g/s)"] / surf_area if surf_area is not None else None 
        data["MassPUA (g/m2)"] = data["Mass (g)"]  / surf_area if surf_area is not None else None 
    else: 
        data["MassPUA (g/m2)"] = None
        data["MLR (g/s)"] = None
        data["MLRPUA (g/s-m2)"] = None
    if "Extinction Area (m2/kg)" not in data.columns:
        data["Extinction Area (m2/kg)"] = None

    test_data = data

######################################################################################################################################################################

############################################### Generate Plot #########################################################                 

    if st.checkbox("Normalize Data Per Unit Area"):
        options = [
            'HRRPUA (kW/m2)','MassPUA (g/m2)', "MLRPUA (g/s-m2)", "THRPUA (MJ/m2)", 
            "MFR (kg/s)", "O2 (Vol fr)", "CO2 (Vol fr)", "CO (Vol fr)", 
            "K Smoke (1/m)", "Extinction Area (m2/kg)"
        ]
        default_value = 'HRRPUA (kW/m2)'
        default_index = options.index(default_value) if default_value in options else 0
        columns_to_graph = st.selectbox(
            "Select data to graph across tests",
            options=options,
            index=0
        )
    else:
        options = [
            'HRR (kW)','Mass (g)', "MLR (g/s)",  "THR (MJ)", 
            "MFR (kg/s)", "O2 (Vol fr)", "CO2 (Vol fr)", "CO (Vol fr)", 
            "K Smoke (1/m)", "Extinction Area (m2/kg)"
        ]
        default_value = 'HRR (kW)'
        default_index = options.index(default_value) if default_value in options else 0
        columns_to_graph = st.selectbox(
            "Select data to graph across tests",
            options=options,
            index=0
        )
        
    if len(columns_to_graph) != 0:
        # Add number inputs for x- and y-axis limits

        x_min = 0
        x_max = np.max(test_data['Time (s)'])
        y_min = np.min(test_data[columns_to_graph])
        y_max = np.max(test_data[columns_to_graph])


        # Create a dictionary to store cutoff values for each column
        column_cutoff_ranges = {}

        st.sidebar.header("Data Modification")
        st.sidebar.write("Enter the time range of the data you woulld like to remove. \n Selections are inclusive.")

        cutoff_start = st.sidebar.number_input("Cut Off Data From Time (s)", value = -1)
        cutoff_end = st.sidebar.number_input("Cut Off Data To Time (s)",value=-1)
        column_cutoff_ranges = (cutoff_start, cutoff_end)
        # Process the data: replace values after the cutoff with NaN for each column
        data_copy = test_data.copy()  # Make a copy to modify the data
        for column in data_copy.columns:
        # Apply the cutoff for each column separately
            cutoff_start, cutoff_end = column_cutoff_ranges
            cutoff_index = ((test_data['Time (s)'] >= cutoff_start) & (test_data['Time (s)'] <= cutoff_end))  
            data_copy.loc[cutoff_index, column] = float('nan')
        # Create plots
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=data_copy['Time (s)'],
                y=data_copy[columns_to_graph],
                name=f"{test_selection} - {columns_to_graph}",
            )
        )
        fig.update_layout(
            yaxis_title=columns_to_graph,
            xaxis_title="Time (s)",
            xaxis_range=[x_min, x_max],
            yaxis_range=[y_min, y_max]
        )
        st.plotly_chart(fig)
######################################################################################################
    
################################################ Saving Adjusted/ Clipped Data ########################################################################
        # Save the adjusted data to a specified path
        st.sidebar.markdown("This button only saves data clipping, csv file modifications are saved seperatley")
        if st.sidebar.button("Save Clipped Data"):
            save_path = str(data_selection)
            save_dir = Path(save_path).parent
            save_dir.mkdir(parents=True, exist_ok=True)
            min_cols = ["Time (s)","Mass (g)","HRR (kW)", "MFR (kg/s)","O2 (Vol fr)", "CO2 (Vol fr)","CO (Vol fr)","K Smoke (1/m)"]
            data_out = data_copy[min_cols].copy()
            if data_out["Mass (g)"].isnull().all():
                if data_copy["MLR (g/s)"].isnull().all():
                    if not data_copy["MLRPUA (g/s-m2)"].isnull().all():
                        data_out["MLRPUA (g/s-m2)"] = data_copy["MLRPUA (g/s-m2)"].copy()
                else:
                    data_out["MLR (g/s)"] = data_copy["MLR (g/s)"].copy()
            if data_out["HRR (kW)"].isnull().all():
                if not data_copy["HRRPUA (kW/m2)"].isnull().all():
                    data_out["HRRPUA (kW/m2)"] = data_copy["HRRPUA (kW/m2)"].copy()
                        # Remove rows where all' values in the selected columns are NaN

            data_out.to_csv(
                save_path,
                float_format="%.4e",
                index=False,
            )

            st.sidebar.success(f"Data saved to {save_path}.")
            #adjust metadata
            with open(test_selection, 'r') as f:
                metadata = json.load(f)
            if cutoff_start != -1 and cutoff_end !=-1:
                test_metadata["Data Corrections"].append(f"{date}: Data from {cutoff_start}s to {cutoff_end}s was removed")
                test_metadata['Manually Prepared'] = date
                with open(test_selection, "w") as f:
                    json.dump(test_metadata, f, indent=4)
            
        if st.checkbox("Modify CSV File"):
            raw_data = pd.read_csv(data_selection)
            view = st.data_editor(raw_data)
            change = st.text_input("Enter Data Correction Performed")
            if st.button("Save Modified CSV"):
                if change:
                    save_path = str(data_selection)
                    save_dir = Path(save_path).parent
                    save_dir.mkdir(parents=True, exist_ok=True)
                    view.to_csv(
                        save_path,
                        float_format="%.4e",
                        index=False,
                    )

                    st.success(f"Data saved to {save_path}.")
                    with open(test_selection, 'r') as f:
                        metadata = json.load(f)
                    test_metadata["Data Corrections"].append(f"{date}: {change}")
                    test_metadata['Manually Prepared'] = date
                    with open(test_selection, "w") as f:
                        json.dump(test_metadata, f, indent=4)
                    st.success(f"Metadata for {test_selection} updated.")
                else:
                    st.warning("Please enter the change performed before saving")



st.sidebar.divider()
st.divider()
st.subheader("Metadata Adjustment")
# region load_metadata
# cache the metadata for faster loading (see here: https://docs.streamlit.io/get-started/fundamentals/advanced-concepts#caching)
@st.cache_data(show_spinner=False)
def load_metadata(show_bar=False):
    with open(test_selection,'r') as f:
        metadata = json.load(f)
    if "Material ID" not in metadata:
        metadata["Material ID"] = None
    fmt_meta = {}
    for key, value in metadata.items():
        if isinstance(value, list):
            fmt_meta[key] = str(value)
        else:
            fmt_meta[key] = value  # or ', '.join(map(str, value)) for a cleaner look
    df = pd.DataFrame.from_dict(fmt_meta, orient='index')
    return df, metadata



def restore_types(edited_df, original_metadata):
    restored = {}
    edited_metadata = df.iloc[:, 0].to_dict()
    for key in edited_metadata:
        original_val = original_metadata.get(key)
        edited_val = edited_metadata.get(key)
        orig_type = type(original_val)

        # Handle None/null
        if original_val is None:
            # If edited value is empty string or looks like "None", treat as None
            if edited_val in ["", None, "None", "null"]:
                restored[key] = None
            else:
                restored[key] = edited_val

        # Integer
        elif orig_type is int:
            try:
                restored[key] = int(edited_val)
            except Exception:
                restored[key] = edited_val

        # Float
        elif orig_type is float:
            try:
                restored[key] = float(edited_val)
            except Exception:
                restored[key] = edited_val

        # Boolean
        elif orig_type is bool:
            # Accept booleans or "True"/"False"
            if isinstance(edited_val, bool):
                restored[key] = edited_val
            elif str(edited_val).lower() in ['true', '1']:
                restored[key] = True
            elif str(edited_val).lower() in ['false', '0']:
                restored[key] = False
            else:
                restored[key] = edited_val

        # List
        elif orig_type is list:
            # Try to safely evaluate string lists
            if isinstance(edited_val, list):
                restored[key] = edited_val
            else:
                # Attempt to convert from string to list
                try:
                    import ast
                    restored[key] = ast.literal_eval(edited_val)
                except:
                    restored[key] = edited_val  # fallback

        # (Add more types as needed, e.g., dict)

        # String (fallback)
        else:
            restored[key] = str(edited_val)

    return restored

# region save_metadata
def save_metadata(df, original_metadata):
    metadata_save = restore_types(df,original_metadata)
    with open(test_selection, 'w') as f:
        json.dump(metadata_save, f, indent=4)


def refresh_meta():
    '''''
    Function activated when the user clicks the "Reload" button.
    This function will clear the cache and reload the metadata
    '''
    st.cache_data.clear()
    df, ogmeta = load_metadata()
    return df, ogmeta

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

df, ogmeta = refresh_meta()

# Override df if reload was requested
if st.session_state.use_reload_df:
    df = st.session_state.reload_df[0].copy()
    st.session_state.use_reload_df = False

def revert_to_parsed():
    "Pulls in the unmodified, parsed JSON file, undoing any saved and unsaved changes that were made"
    metadata = json.load(open(metadata_path_map[selected_test]))
    ogform = metadata["Original Source"]
    parsed_path = str(PARSED_METADATA_PATH) + f"\\{ogform}"
    bad_data = str(metadata_path_map[selected_test])
    original_data = bad_data.replace(str(INPUT_DATA_PATH), parsed_path)
    save_path = str(metadata_path_map[selected_test])
    shutil.copy(original_data, save_path)
    st.success(f"Metadata for {selected_test} reverted to parsed")

# region delete_files
def delete_files():
    # delete the metadata files as well as their corresponding csv files
    #tag parsed file as being bad so it is not re-imported
    active_file = metadata_path_map[selected_test]
    metadata = json.load(open(active_file))
    ogform = metadata["Original Source"]
    parsed_path = str(PARSED_METADATA_PATH) + f"\\{ogform}"
    parsed_file = Path(str(active_file).replace(str(INPUT_DATA_PATH), parsed_path))
    with open(parsed_file, "r", encoding="utf-8") as w:  
        parsed = json.load(w) 
    parsed["Bad Data"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open (parsed_file, "w", encoding="utf-8") as w:
        json.dump(parsed, w, indent = 4)
    metadata_path_map[selected_test].unlink()
    metadata_path_map[selected_test].with_suffix(".csv").unlink()
    # clear the cache so that the metadata files are reloaded
    st.cache_data.clear()
    st.success(f"{selected_test} files deleted")



# --- Store original copy for reload ---
if 'df_original' not in st.session_state:
    st.session_state.df_original = df.copy()
st.session_state.df = df


# sidebar UI
st.button("Reload Metadata", on_click= reload_metadata, use_container_width=True)

if test_selection:
    df = st.data_editor(
        df,
        key=st.session_state.editor_key,  
        use_container_width=True,
    )

st.button("Save Metadata", on_click=lambda: save_metadata(df,ogmeta), use_container_width=True)
st.button("Revert Metadata", on_click= revert_to_parsed, use_container_width=True)
st.button("Delete files", on_click=delete_files, use_container_width=True)

# region export_metadata
def export_metadata(df, original_metadata):
    print('indafunction')
    row =  restore_types(df,original_metadata)
    with open(test_selection,'r') as f:
        metadata = json.load(f)

        # Find any difference (value mismatch) between the input and restored row
    for key in metadata:
        v1 = metadata.get(key)
        v2 = row.get(key)
        if v1 != v2:
            st.warning(f"Export Aborted: Difference detected for key '{key}'.\nOriginal: {v1}\nEdited: {v2}. Please save or reload the metadata prior to export")
            return  # End the function after the first difference    
    
    if metadata.get("Material ID") is None or metadata.get("Material ID") in ["nan", ""]:
        st.warning(f"Export Aborted: Please enter a valid Material ID")
        return
    
    if (metadata.get("Heat Flux (kW/m2)") is None) or metadata.get("Heat Flux (kW/m2)") == "Not found":
        st.warning(f"Export Aborted: Please enter an Heat Flux")
        return
    ogform = metadata["Original Source"]
    date = metadata["Test Date"]
    dt_obj = None
    if isinstance(date, datetime):
        dt_obj = date
    else:
        formats = ["%d %b %Y","%d %b %y","%m/%d/%y","%m/%d/%Y","%Y-%m-%d","%Y-%m-%dT%H:%M:%S"]
        for fmt in formats:
            try:
                dt_obj = datetime.strptime(str(date), fmt)  # Ensure string
                break  # Parsing succeeded
            except (ValueError, TypeError):
                continue

    if dt_obj is None:
        st.warning(f"Export Aborted: Unrecognized date format: {date}.")
        return

    metadata['Test Date'] = dt_obj.strftime("%Y-%m-%d")
    metadata['SmURF'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    parsed_path = str(PARSED_METADATA_PATH) + f"\\{ogform}"
    prepared_path = str(PREPARED_METADATA_PATH)+ f"\\{ogform}"
    prepared_data =  str(PREPARED_DATA_PATH)+ f"\\{ogform}"
    if not OUTPUT_DATA_PATH.exists():
        OUTPUT_DATA_PATH.mkdir(parents=True, exist_ok=True)
    if not Path(prepared_path).exists():
        Path(prepared_path).mkdir(parents=True, exist_ok=True)
    if not Path(prepared_data).exists():
        Path(prepared_data).mkdir(parents=True, exist_ok=True)
    material_id = metadata.get("Material ID")    
    
    filename_parts = [
        material_id,
        str(int(metadata["Heat Flux (kW/m2)"])),
        "vert" if metadata["Orientation"] == "VERTICAL" else "horiz",
    ]

    # if there is a specimen number, include that in the filename
    if metadata.get("Specimen Number") is not None and metadata.get("Specimen Number") != "":
        filename_parts.insert(3, f"{metadata['Specimen Number']}")
    
    # join all the filename parts together with a dash & add the file extension (.json)
    metadata['Testname'] = "_".join(filename_parts)
    new_filename = "_".join(filename_parts) + ".json"
    old_filename = metadata["Original Testname"] + '.json'
    neworder = ['Material ID', 'Sample Mass (g)','Specimen Number','Testname', 
            'Instrument', 'Test Date', 'Institution','Preparsed','Parsed','Auto Prepared', 'Manually Prepared', "SmURF", 'Autoprocessed', 
                'Manually Reviewed Series','Pass Review', 'Published', "Original Testname", "Bad Data", "Original Source", "Heat Flux (kW/m2)", 'Orientation', 'Material Name','Conversion Factor', 'C Factor',
                'Surface Area (m2)','Time to Ignition (s)', 'Residual Mass (g)', 'Residue Yield (g/g)','Mass Consumed', "Soot Average (g/g)",
                'Peak Heat Release Rate (kW/m2)', 'Peak Mass Loss Rate (g/s-m2)', 'Comments', 'Data Corrections' ]
    reordered_metadata = {key: metadata[key] for key in neworder}
    for key in metadata:
        if key not in neworder:
            reordered_metadata[key] = metadata[key]

    
    # save the metadata file, including to parsed
    with open(OUTPUT_DATA_PATH / new_filename, "w") as f:
        json.dump(reordered_metadata, f, indent=4)
    with open(Path(prepared_path) / new_filename, "w") as f:
        json.dump(reordered_metadata, f, indent=4)
    with open(Path(parsed_path) / old_filename, "w") as f:
        json.dump(reordered_metadata, f, indent=4)
    with open(INPUT_DATA_PATH / old_filename, "w") as f:
        json.dump(reordered_metadata, f, indent=4)

            # Get the CSV data file as well and save that in the same folder as the metadata file
    data = pd.read_csv(data_selection)

    #Create minimum columns if they can be made (ex: found the surface area of a test)
    if "HRRPUA (kW/m2)" in data.columns and reordered_metadata["Surface Area (m2)"]:
        data["HRR (kW)"] = data["HRRPUA (kW/m2)"] * reordered_metadata["Surface Area (m2)"]
        data.drop("HRRPUA (kW/m2)", inplace=True, axis = 1)
    if "MassPUA (g/m2)" in data.columns and reordered_metadata["Surface Area (m2)"]:
        data["Mass (g)"] = data["MassPUA (g/m2)"] * reordered_metadata["Surface Area (m2)"]
        data.drop("MassPUA (g/m2)", inplace=True, axis = 1)
    elif "Mass LossPUA (g/m2)" in data.columns and reordered_metadata["Surface Area (m2)"]:
        data["Mass Loss (g)"] = data["Mass LossPUA (g/m2)"] * reordered_metadata["Surface Area (m2)"]
        data.drop("Mass LossPUA (g/m2)", inplace=True, axis = 1)
    elif "MLRPUA (g/s-m2)" in data.columns and reordered_metadata["Surface Area (m2)"]:
        data["MLR (g/s)"] = data["MLRPUA (g/s-m2)"] * reordered_metadata["Surface Area (m2)"]
        data.drop("MLRPUA (g/s-m2)", inplace=True, axis = 1)
    if "Mass Loss (g)" in data.columns and reordered_metadata["Sample Mass (g)"]:
        data["Mass (g)"] = reordered_metadata["Sample Mass (g)"] - data["Mass Loss (g)"]  
        data.drop("Mass Loss (g)", inplace=True, axis = 1)
    
    max_column_order = ["Time (s)", "Mass (g)", "HRR (kW)", "MFR (kg/s)","O2 (Vol fr)", "CO2 (Vol fr)","CO (Vol fr)",
                        "K Smoke (1/m)","Extinction Area (m2/kg)", "Mass Loss (g)", "Mass LossPUA (g/m2)", "MLR (g/s)", "MLRPUA(g/s-m2)",
                        "HRRPUA (kW/m2)"]
    reordered_data = pd.DataFrame()
    for c in max_column_order:
        if c in data.columns:
            reordered_data[c] = data[c]
    
    reordered_data.to_csv(Path(prepared_data) / new_filename.replace(".json", ".csv"), index=False)
    reordered_data.to_csv(OUTPUT_DATA_PATH/ new_filename.replace(".json", ".csv"), index=False)
    st.success(f"Data and Metadata for {new_filename}")
 



st.sidebar.button("Export Data and Metadata", on_click=lambda: export_metadata(df,ogmeta), use_container_width=True)
st.sidebar.markdown("Selected files are renamed, and their data and metadata are exported to the prepared stage")


st.divider()


st.markdown("#### Notes")
readme = SCRIPT_DIR / "README.md"
section_title = "### SmURF Editor"

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

