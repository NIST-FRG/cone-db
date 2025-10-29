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
        df=pd.read_csv(test_name_map[test_stem])
        test_metadata = (json.load(open(metadata_name_map[test_stem])))
        surf_area = test_metadata.get("Surface Area (m2)")
        flux = test_metadata.get("Heat Flux (kW/m2)")
        if flux != None :
            df["t * EHF (kJ/m2)"] = df["Time (s)"] * flux
        else:
            df["t * EHF (kJ/m2)"] = None
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

        test_data.append(df)
######################################################################################################################################################

########################################### Generate Plots ###################################################################################      
    x_axis_column = st.selectbox(
        "Select the column for the x-axis",
        options=['Time (s)', 't * EHF (kJ/m2)'],
    )
        
    if st.checkbox("Normalize Data Per Unit Area"):
        y_axis_columns = st.multiselect(
        "Select data to graph across tests",
        options= ['MassPUA (g/m2)',"MLRPUA (g/s-m2)",'HRRPUA (kW/m2)', "THRPUA (MJ/m2)", 
                  "MFR (kg/s)","O2 (Vol fr)", "CO2 (Vol fr)","CO (Vol fr)", "K Smoke (1/m)", "Extinction Area (m2/kg)"]
    )
    else:     
        # Select which column(s) to graph
        y_axis_columns = st.multiselect(
            "Select data to graph across tests",
            options= ['Mass (g)',"MLR (g/s)",'HRR (kW)',"THR (MJ)", 
                 "MFR (kg/s)","O2 (Vol fr)", "CO2 (Vol fr)","CO (Vol fr)", "K Smoke (1/m)", "Extinction Area (m2/kg)"]
    )
        

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
