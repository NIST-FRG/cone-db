import json
import re
import shutil
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from const import INPUT_DATA_PATH, PARSED_DATA_PATH
from scipy.signal import savgol_filter

################################ Title of Page #####################################################
st.set_page_config(page_title="Cone Data Editor", page_icon="📈", layout="wide")
st.title("Cone Data Editor")

#####################################################################################################
############################## Get test files, select by material, then material id, then test #################
# Get the paths to all the test files
test_name_map = {p.stem: p for p in list(INPUT_DATA_PATH.rglob("*.csv"))}

# Get the paths to all the metadata files
metadata_name_map = {p.stem: p for p in list(INPUT_DATA_PATH.rglob("*.json"))}

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
        parsed_data = str(test_name_map[test_selection])
        original_data = parsed_data.replace(str(INPUT_DATA_PATH), str(PARSED_DATA_PATH))
        save_path = str(test_name_map[test_selection])
        shutil.copy(original_data, save_path)
        test_metadata["Data Corrections"].append(f"{date}: Data reverted to original")
        with open(metadata_name_map[test_selection], "w") as f:
            json.dump(test_metadata, f, indent=4)
        st.sidebar.success(f"Data in {save_path} reverted to original.")
        
    df=pd.read_csv(test_name_map[test_selection])
    surf_area = test_metadata["Surface Area (m2)"]
    df["HRRPUA (kW/m2)"] = df["HRR (kW)"]/ surf_area
    df['dt'] = df["Time (s)"].diff()
    df['Q (MJ)'] = (df['HRR (kW)']*df['dt'])/1000
    df['THR(MJ)'] = df["Q (MJ)"].cumsum()
    df['THRPUA (MJ/m2)'] = df["THR(MJ)"]/surf_area
    if "Mass (g)" in df.columns:
        df["MassPUA (g/m2)"] = df["Mass (g)"]/surf_area
        df['MLR (g/s)'] = np.gradient(df["Mass (g)"]) # COME BACK TO THIS
    else:
        df["Mass (g)"] = None
        df["MassPUA (g/m2)"] = None
    df['MLRPUA (g/s-m2)'] = df['MLR (g/s)']/surf_area
    test_data = df
######################################################################################################################################################################

############################################### Generate Plot #########################################################                 

    if st.checkbox("Normalize Data Per Unit Area"):
        columns_to_graph = st.multiselect(
        "Select data to graph across tests",
        options= ['MassPUA (g/m2)',"MLRPUA (g/s-m2)",'HRRPUA (kW/m2)', "THRPUA (MJ/m2)"]
    )
    else:     
        # Select which column(s) to graph
        columns_to_graph = st.multiselect(
            "Select data to graph across tests",
            options= ['Mass (g)',"MLR (g/s)",'HRR (kW/m2)',"THR (MJ/m2)"]
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

        cutoff_start = st.sidebar.number_input("Cut Off Data From Time (s)", value = 0)
        cutoff_end = st.sidebar.number_input("Cut Off Data To Time (s)",value=0)
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
        if st.sidebar.button("Save Adjusted Data"):
            save_path = str(test_name_map[test_selection])
            save_dir = Path(save_path).parent
            save_dir.mkdir(parents=True, exist_ok=True)
            data_out = data_copy[['Time (s)', 't * EHF (kJ/m2)', 'HRR (kW/m2)', "MLR (g/s-m2)", "THR (MJ/m2)"]]
            # Remove rows where all values in the selected columns are NaN
            data_out = data_out.dropna()
            data_out.to_csv(
                save_path,
                float_format="%.4e",
                index=False,
            )

            st.sidebar.success(f"Data saved to {save_path}.")
            #adjust metadata
            with open(metadata_name_map[test_selection], 'r') as f:
                metadata = json.load(f)
            test_metadata["Data Corrections"].append(f"{date}: Data from {cutoff_start} s to {cutoff_end} s was removed")
            test_metadata['Manually Prepared'] = date
            with open(metadata_name_map[test_selection], "w") as f:
                json.dump(test_metadata, f, indent=4)
            
            
