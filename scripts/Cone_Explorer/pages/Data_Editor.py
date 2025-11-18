import json
import re
import shutil
from datetime import datetime
from pathlib import Path
import sys
import numpy as np
import pandas as pd
 
import streamlit as st
from scipy.signal import savgol_filter
import plotly.graph_objects as go

PROJECT_ROOT = Path(__file__).resolve().parents[2]  # .../Scripts
print(PROJECT_ROOT)
sys.path.append(str(PROJECT_ROOT))
from Cone_Explorer.const import (
    PREPARED_DATA_PATH,
    PREPARED_METADATA_PATH,
    SCRIPT_DIR
)



################################ Title of Page #####################################################
st.set_page_config(page_title="Cone Data Editor", page_icon="📈", layout="wide")
st.title("Cone Data Editor")

#####################################################################################################
############################## Get test files, select by material, then material id, then test #################
metadata_name_map = {p.stem: p for p in list(PREPARED_METADATA_PATH.rglob("*.json"))}
test_name_map = {p.stem: p for p in list(PREPARED_DATA_PATH.rglob("*.csv"))}

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
        prepared_data = str(test_name_map[test_selection])
        og_name = test_metadata["Original Testname"]
        name = test_metadata["Testname"]
        original_data = prepared_data.replace("Exp-Data_Prepared-Final", "Exp-Data_Parsed").replace(name,og_name)
        save_path = str(test_name_map[test_selection])
        shutil.copy(original_data, save_path)
        test_metadata["Data Corrections"].append(f"{date}: Data reverted to original")
        with open(metadata_name_map[test_selection], "w") as f:
            json.dump(test_metadata, f, indent=4)
        st.sidebar.success(f"Data in {save_path} reverted to original.")
        
    data = pd.read_csv(test_name_map[test_selection])
    surf_area = test_metadata.get("Surface Area (m2)")
    mass = test_metadata.get("Sample Mass (g)")
    X_O2_i = test_metadata.get("X_O2 Initial")
    X_CO2_i = test_metadata.get("X_CO2 Initial")
    X_CO_i = test_metadata.get("X_CO Initial")
    rel_humid = test_metadata.get('Relative Humidity (%)')
    amb_temp = test_metadata.get("Ambient Temperature (°C)")
    amb_pressure = test_metadata.get("Barometric Pressure (Pa)")
    duct_diam = test_metadata.get("Duct Diameter (m)")
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
        data['MLR (g/s)'] = None
        m = data["Mass (g)"]
        for i in range(len(data)):
            if i == 0:
                data.loc[i,"MLR (g/s)"] = (25*m[0] - 48*m[1] + 36*m[2] - 16*m[3] + 3*m[4])/(12*data['dt'].iloc[i])
            elif i == 1:
                data.loc[i,"MLR (g/s)"] = (3*m[0] + 10*m[1] - 18*m[2] + 6*m[3] - m[4])/(12*data['dt'].iloc[i])
            elif i ==len(data) -2:
                data.loc[i,"MLR (g/s)"] = (-3*m[i+1] - 10*m[i] + 18*m[i-1] - 6*m[i-2] + m[i-3])/(12*data['dt'].iloc[i])
            elif i == len(data)-1:
                data.loc[i,"MLR (g/s)"] = (-25*m[i] + 48*m[i-1] - 36*m[i-2] + 16*m[i-3] - 3*m[i-4])/(12*data['dt'].iloc[i])
            else:
                data.loc[i,"MLR (g/s)"] = (-m[i-2]+ 8*m[i-1]- 8*m[i+1]+m[i+2])/(12*data['dt'].iloc[i])

        data["MLRPUA (g/s-m2)"] = data["MLR (g/s)"] / surf_area if surf_area is not None else None 
        data["MassPUA (g/m2)"] = data["Mass (g)"]  / surf_area if surf_area is not None else None 
    else: 
        data["MassPUA (g/m2)"] = None
        data["MLR (g/s)"] = None
        data["MLRPUA (g/s-m2)"] = None

    #weight air taken from 2077, this publication also used ambient pressure in the building, so will I: combustion products/humidty negligable
    W_air = 28.97
    data['Rho_Air (kg/m3)'] = ((amb_pressure/1000) * W_air)  / (8.314 * data['T Duct (K)'])
    data["V Duct (m3/s)"] = data['MFR (kg/s)'] / data["Rho_Air (kg/m3)"]
    data["Extinction Area (m2/kg)"] = (data['V Duct (m3/s)'] * data['K Smoke (1/m)']) / (data['MLR (g/s)']/1000)
    #Finding Soot  production based on FCD User Guide- but bring area into eq so have Vduct
    #Says to use smoke production sigmas = 8.7m2/g, not sigmaf
    data['Soot Production (g/s)'] = 1/8.7 * data["K Smoke (1/m)"] * data['V Duct (m3/s)']
    data['Smoke Production (m2/s)'] = data['K Smoke (1/m)'] * data['V Duct (m3/s)']
    
    data["HoC (MJ/kg)"] = data["HRRPUA (kW/m2)"] / data["MLRPUA (g/s-m2)"]
    # Grab values
    HoC_values = data["HoC (MJ/kg)"].to_numpy()

    # Compute z-scores, handling NaNs as needed
    HOCmean = np.nanmean(HoC_values)
    HOCstd = np.nanstd(HoC_values)
    HOCz = (HoC_values - HOCmean) / HOCstd

    # Build mask for outliers or negatives
    mask = (HoC_values < 0) | (np.abs(HOCz) > 2)

    # Assign zero where mask is True
    data.loc[mask, "HoC (MJ/kg)"] = 0

    ## Gas Production and Yield
    W_CO2 = 44.01
    W_CO = 28.01
    W_O2 = 32
    #Production and yields calculated by following FCD user guide
    data['CO2 Production (g/s)'] = (W_CO2/W_air) * (data['CO2 (Vol fr)'] - X_CO2_i) * data['MFR (kg/s)'] *1000
    data['CO Production (g/s)'] = (W_CO/W_air) * (data['CO (Vol fr)'] - X_CO_i) * data['MFR (kg/s)'] *1000
    data['O2 Consumption (g/s)'] = (W_O2/W_air) * (X_O2_i - data['O2 (Vol fr)'] ) * data['MFR (kg/s)'] *1000
    test_data = data

######################################################################################################################################################################

############################################### Generate Plot #########################################################                 

    if st.checkbox("View Additional Calculated Properties"):
        options = [
            'HRRPUA (kW/m2)','MassPUA (g/m2)', "MLRPUA (g/s-m2)", "THRPUA (MJ/m2)", 
           "CO2 Production (g/s)", "CO Production (g/s)", "O2 Consumption (g/s)", 
           "Soot Production (g/s)", "K Smoke (1/m)","MFR (kg/s)",'V Duct (m3/s)' ]
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
             "O2 (Vol fr)", "CO2 (Vol fr)", "CO (Vol fr)", 'Smoke Production (m2/s)']
        
        default_value = 'HRR (kW)'
        default_index = options.index(default_value) if default_value in options else 0
        columns_to_graph = st.selectbox(
            "Select data to graph across tests",
            options=options,
            index=0
        )
    st.caption('Legacy Data May Be Missing Several Data Columns')
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
########################################################################################################################################
    
################################################ Saving Adjusted/ Clipped Data ########################################################################
        # Save the adjusted data to a specified path
        st.sidebar.markdown("This button only saves data clipping, csv file modifications are saved seperatley")
        if st.sidebar.button("Save Clipped Data"):
            save_path = str(test_name_map[test_selection])
            save_dir = Path(save_path).parent
            save_dir.mkdir(parents=True, exist_ok=True)
            min_cols = ["Time (s)","Mass (g)","HRR (kW)", "MFR (kg/s)","T Duct (K)","O2 (Vol fr)", "CO2 (Vol fr)","CO (Vol fr)","K Smoke (1/m)"]
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
            test_metadata["Data Corrections"].append(f"{date}: Data from {cutoff_start}s to {cutoff_end}s was removed")
            test_metadata['Manually Prepared'] = date
            with open(metadata_name_map[test_selection], "w") as f:
                json.dump(test_metadata, f, indent=4)
            
 
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
