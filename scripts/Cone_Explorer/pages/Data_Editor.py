import json
import re
import shutil
from datetime import datetime
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from scipy.signal import savgol_filter

PROJECT_ROOT = Path(__file__).resolve().parents[2]  # .../Scripts
print(PROJECT_ROOT)
sys.path.append(str(PROJECT_ROOT))
from Cone_Explorer.const import (
    INPUT_DATA_PATH,
    PARSED_DATA_PATH, 
    SCRIPT_DIR
)



################################ Title of Page #####################################################
st.set_page_config(page_title="Cone Data Editor", page_icon="📈", layout="wide")
st.title("Cone Data Editor")

#####################################################################################################
############################## Get test files, select by material, then material id, then test #################
# Get the paths to all the test files
test_name_map = {p.stem: p for p in list(INPUT_DATA_PATH.rglob("*.csv"))}
# Get the paths to all the metadata files
metadata_name_map = {p.stem: p for p in list(INPUT_DATA_PATH.rglob("*.json"))}

if st.checkbox('SmURF Filter'):
    st.write("Select the test status you would like to view")
    test_types = ['SmURFed', "Not SmURFed"]
    selected_type = st.selectbox("Choose SmURF status", test_types)
    # Filter tests based on selected types
    filtered_tests = []
    if selected_type == 'SmURFed':
        for test_name, test_value in metadata_name_map.items():     
            with open(test_value, 'r') as f:
                metadata = json.load(f)
            if metadata["SmURF"] != None:
                filtered_tests.append(test_name)
    else:
        for test_name, test_value in metadata_name_map.items():     
            with open(test_value, 'r') as f:
                metadata = json.load(f)
            if metadata["SmURF"] == None:
                filtered_tests.append(test_name)
    metadata_name_map = {test: metadata_name_map[test] for test in filtered_tests if test in metadata_name_map}
    test_name_map = {test: metadata_name_map[test].with_suffix('.csv') for test in filtered_tests}

def get_markdown_number(key):
    # Example key: "test0477_0857_001"
    match = re.search(r'_(\d+)_\d+', key)
    return int(match.group(1)) if match else 0


if st.checkbox("Sort by Markdown File Number"):
    sortedkeys = sorted(test_name_map.keys(), key = get_markdown_number)
    test_selection = st.selectbox(
        "Select a test to view and edit:",
        options=sortedkeys,
    )
else:
    test_selection = st.selectbox(
        "Select a test to view and edit:",
        options=test_name_map.keys(),
    ) 
#########################################################################################################

###################### Generate Dataframe to Plot from Each Test Selected####################################      
date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
if test_selection:
    test_metadata = json.load(open(metadata_name_map[test_selection]))
    ######## Revert data in case it was clipped and you want the full dataframe from parsed back#####################
    if st.sidebar.button("Revert to Parsed Data"):
        ogform = test_metadata["Original Source"]
        explorer_data = str(test_name_map[test_selection])
        parsed_path = str(PARSED_DATA_PATH) + f"\\{ogform}"
        original_data = explorer_data.replace(str(INPUT_DATA_PATH), parsed_path)
        save_path = str(test_name_map[test_selection])
        shutil.copy(original_data, save_path)
        test_metadata["Data Corrections"].append(f"{date}: Data reverted to original")
        with open(metadata_name_map[test_selection], "w") as f:
            json.dump(test_metadata, f, indent=4)
        st.sidebar.success(f"Data in {save_path} reverted to original.")
        
    df=pd.read_csv(test_name_map[test_selection])
    surf_area = test_metadata.get("Surface Area (m2)")
    mass = test_metadata.get("Sample Mass (g)")
    df['dt'] = df["Time (s)"].diff()
    #Normal and area adjusted HRR and THR generation
    if "HRRPUA (kW/m2)" not in df.columns:
        df["HRRPUA (kW/m2)"] = df["HRR (kW)"] / surf_area if surf_area is not None else None
    df["QPUA (MJ/m2)"] = (df['HRRPUA (kW/m2)']*df['dt'])/1000
    df["THRPUA (MJ/m2)"] = df["QPUA (MJ/m2)"].cumsum()
    df['Q (MJ)'] = (df['HRR (kW)']*df['dt'])/1000
    df['THR (MJ)'] = df["Q (MJ)"].cumsum()
   

   #Mass and Mass Loss Rate Data
    if "MLR (g/s)" in df.columns:
        df["MLRPUA (g/s-m2)"] = df["MLR (g/s)"] / surf_area if surf_area is not None else None
        df["MassPUA (g/m2)"] = None
    elif "MLRPUA (g/s-m2)" in df.columns:
        df["MLR (g/s)"] = df["MLRPUA (g/s-m2)"] / surf_area if surf_area is not None else None
        df["MassPUA (g/m2)"] = None
    elif not df["Mass (g)"].isnull().all():
        df['MLR (g/s)'] = abs(np.gradient(df["Mass (g)"])) # COME BACK TO THIS
        df["MLRPUA (g/s-m2)"] = df["MLR (g/s)"] / surf_area if surf_area is not None else None 
        df["MassPUA (g/m2)"] = df["Mass (g)"]  / surf_area if surf_area is not None else None 
    else: 
        df["MassPUA (g/m2)"] = None
        df["MLR (g/s)"] = None
        df["MLRPUA (g/s-m2)"] = None
    if "Extinction Area (m2/kg)" not in df:
        df["Extinction Area (m2/kg)"] = None

    test_data = df

######################################################################################################################################################################

############################################### Generate Plot #########################################################                 

    if st.checkbox("Normalize Data Per Unit Area"):
        columns_to_graph = st.multiselect(
        "Select data to graph across tests",
        options= ['MassPUA (g/m2)',"MLRPUA (g/s-m2)",'HRRPUA (kW/m2)', "THRPUA (MJ/m2)", 
                  "MFR (kg/s)","O2 (Vol fr)", "CO2 (Vol fr)","CO (Vol fr)", "K Smoke (1/m)", "Extinction Area (m2/kg)"]
    )
    else:     
        # Select which column(s) to graph
        columns_to_graph = st.multiselect(
            "Select data to graph across tests",
            options= ['Mass (g)',"MLR (g/s)",'HRR (kW)',"THR (MJ)", 
                 "MFR (kg/s)","O2 (Vol fr)", "CO2 (Vol fr)","CO (Vol fr)", "K Smoke (1/m)", "Extinction Area (m2/kg)"]
    )
    
    if len(columns_to_graph) != 0:
        # Add number inputs for x- and y-axis limits
        st.sidebar.header("Axis Range Controls")
        x_min_number = 0
        x_max_number = np.max(test_data['Time (s)'])
        y_min_number = np.min(test_data[columns_to_graph])
        y_max_number = np.max(test_data[columns_to_graph])

        x_min = st.sidebar.number_input("X-axis min", value=float(x_min_number))
        x_max = st.sidebar.number_input("X-axis max", value=float(x_max_number))
        y_min = st.sidebar.number_input("Y-axis min", value=float(y_min_number))
        y_max = st.sidebar.number_input("Y-axis max", value=float(y_max_number))


        # Create a dictionary to store cutoff values for each column
        column_cutoff_ranges = {}

        st.sidebar.header("Cutoff Values")
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
        for column in columns_to_graph:
            fig.add_trace(
                go.Scatter(
                x=data_copy['Time (s)'],
                y=data_copy[column],
                    name=f"{test_selection} - {column}",
                    )
                )

        fig.update_layout(
            yaxis_title=columns_to_graph[0] if columns_to_graph else 'Values', 
            xaxis_title="Time (s)",
            xaxis_range=[x_min, x_max],
            yaxis_range=[y_min, y_max]
        )

    
        st.plotly_chart(fig)
########################################################################################################################################
    
################################################ Saving Adjusted/ Clipped Data ########################################################################
        # Save the adjusted data to a specified path
        st.sidebar.markdown("This button only saves data clipping, csv file modifications are saved seperatley")
        if st.sidebar.button("Save Clipped Data"):
            save_path = str(test_name_map[test_selection])
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
            with open(metadata_name_map[test_selection], 'r') as f:
                metadata = json.load(f)
            test_metadata["Data Corrections"].append(f"{date}: Data from {cutoff_start}s to {cutoff_end}s was removed")
            test_metadata['Manually Prepared'] = date
            with open(metadata_name_map[test_selection], "w") as f:
                json.dump(test_metadata, f, indent=4)
            
        if st.checkbox("Modify CSV File"):
            raw_data = pd.read_csv(test_name_map[test_selection])
            view = st.data_editor(raw_data)
            change = st.text_input("Enter Data Correction Performed")
            if st.button("Save Modified CSV"):
                if change:
                    save_path = str(test_name_map[test_selection])
                    save_dir = Path(save_path).parent
                    save_dir.mkdir(parents=True, exist_ok=True)
                    view.to_csv(
                        save_path,
                        float_format="%.4e",
                        index=False,
                    )

                    st.success(f"Data saved to {save_path}.")
                    with open(metadata_name_map[test_selection], 'r') as f:
                        metadata = json.load(f)
                    test_metadata["Data Corrections"].append(f"{date}: {change}")
                    test_metadata['Manually Prepared'] = date
                    with open(metadata_name_map[test_selection], "w") as f:
                        json.dump(test_metadata, f, indent=4)
                    st.success(f"Metadata for {test_selection} updated.")
                else:
                    st.warning("Please enter the change performed before saving")
    st.markdown("#### Notes")
    readme = SCRIPT_DIR / "README.md"
    section_title = "### Data Editor"

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
