import json
from pathlib import Path
import pandas as pd
import numpy as np
import re
import streamlit as st
import sys
import plotly.express as px
import plotly.graph_objects as go


PROJECT_ROOT = Path(__file__).resolve().parents[2]  # .../Scripts

sys.path.append(str(PROJECT_ROOT))
from Cone_Explorer.const import (
    INPUT_DATA_PATH,
    PARSED_METADATA_PATH,
    PREPARED_METADATA_PATH,
    SCRIPT_DIR, 
    PARSED_DATA_PATH,
    PREPARED_DATA_PATH
)



################################ Title of Page #####################################################
st.set_page_config(page_title="Cone Data Viewer", page_icon="📈", layout="wide")

st.title("Cone Data Viewer")
#####################################################################################################

################### Get test files, select by material, then material id, then test #################

st.write("Select the test status you would like to view")
test_types = ['SmURFed', "Not SmURFed"]
selected_type = st.selectbox("Choose SmURF status", test_types)
if selected_type == "Not SmURFed":
    # Get the paths to all the test metadata files
    metadata_name_map = {p.stem: p for p in list(PARSED_METADATA_PATH.rglob("*.json"))}
    test_name_map = {p.stem: p for p in list(PARSED_DATA_PATH.rglob("*.csv"))}
else:
    # Get the paths to all the test metadata files
    metadata_name_map = {p.stem: p for p in list(PREPARED_METADATA_PATH.rglob("*.json"))}
    test_name_map = {p.stem: p for p in list(PREPARED_DATA_PATH.rglob("*.csv"))}



test_selection = st.multiselect(
    "Select a test to view and edit:",
    options=test_name_map.keys(),
) 
##########################################################################################################
        
################### Generate Dataframe to Plot from Each Test Selected####################################      
if len(test_selection) != 0:
    # Get the data & metadata for each test
    test_data = []
    for i, test_stem in enumerate(test_selection):
        test_metadata = (json.load(open(metadata_name_map[test_stem])))
        data = pd.read_csv(test_name_map[test_stem])
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

        
        data["Extinction Area (m2/kg)"] = (data['V Duct (m3/s)'] * data['K Smoke (1/m)']) / (data['MLR (g/s)']/1000)
        #Finding Soot  production based on FCD User Guide- but bring area into eq so have Vduct
        #Says to use smoke production sigmas = 8.7m2/g, not sigmaf
        data['Soot Production (g/s)'] = 1/8.7 * data["K Smoke (1/m)"] * data['V Duct (m3/s)']
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
        # Calculate ambient XO2 following ASTM 1354 A.1.4.5
        p_sat_water = 6.1078 * 10**((7.5*amb_temp)/(237.3 + amb_temp)) * 100 #saturation pressure in pa, Magnus approx
        p_h2o = rel_humid/100 * p_sat_water
        X_H2O_initial = p_h2o / amb_pressure
        #weight air taken from 2077, this publication also used ambient pressure in the building, so will I
        W_dryair = 28.963
        W_air = X_H2O_initial * 18.02 + (1-X_H2O_initial) * W_dryair
        W_CO2 = 44.01
        W_CO = 28.01
        W_O2 = 32
        #Production and yields calculated by following FCD user guide
        data['CO2 Production (g/s)'] = (W_CO2/W_air) * (data['CO2 (Vol fr)'] - X_CO2_i) * data['MFR (kg/s)'] *1000
        data['CO Production (g/s)'] = (W_CO/W_air) * (data['CO (Vol fr)'] - X_CO_i) * data['MFR (kg/s)'] *1000
        data['O2 Consumption (g/s)'] = (W_O2/W_air) * (X_O2_i - data['O2 (Vol fr)'] ) * data['MFR (kg/s)'] *1000

        test_data.append(data)
######################################################################################################################################################

########################################### Generate Plots ###################################################################################      
    x_axis_column = st.selectbox(
        "Select the column for the x-axis",
        options=['Time (s)', 't * EHF (kJ/m2)'],
    )
        
    if st.checkbox("View Additional Calculated Properties"):
        options = [
            'HRRPUA (kW/m2)','MassPUA (g/m2)', "MLRPUA (g/s-m2)", "THRPUA (MJ/m2)", 
             "Extinction Area (m2/kg)","HoC (MJ/kg)", "CO2 Production (g/s)", 
             "CO Production (g/s)", "O2 Consumption (g/s)", "Soot Production (g/s)"
        ]
        default_value = 'HRRPUA (kW/m2)'
        default_index = options.index(default_value) if default_value in options else 0
        y_axis_columns = st.multiselect(
            "Select data to graph across tests",
            options=options,
        )
    else:
        options = [
            'HRR (kW)','Mass (g)', "MLR (g/s)",  "THR (MJ)", 
            "MFR (kg/s)", "O2 (Vol fr)", "CO2 (Vol fr)", "CO (Vol fr)", 
            "K Smoke (1/m)", 'V Duct (m3/s)'
        ]
        default_value = 'HRR (kW)'
        default_index = options.index(default_value) if default_value in options else 0
        y_axis_columns = st.multiselect(
            "Select data to graph across tests",
            options=options,
        )
    st.caption('Legacy Data May Be Missing Several Data Columns')
        

    # Create plots
    if x_axis_column and y_axis_columns:
        for y_column in y_axis_columns:
            fig = go.Figure()
            # Add additional traces for each test
            for i in range(len(test_data)):
                fig.add_trace(
                    go.Scatter(
                        x=test_data[i][x_axis_column],  # Use selected x-axis column
                        y=test_data[i][y_column],  # Use the current y-axis column
                        name=test_selection[i],
                    )
                )
            
            # Update layout for the current figure
            fig.update_layout(
                yaxis_title=y_column,
                xaxis_title=x_axis_column,
            )
            
            st.markdown(f"#### {y_column} vs {x_axis_column}")
            st.plotly_chart(fig)


    st.markdown("#### Notes")
    readme = SCRIPT_DIR / "README.md"
    section_title = "### Compare Tests"

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
