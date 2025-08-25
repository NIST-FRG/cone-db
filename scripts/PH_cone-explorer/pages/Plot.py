import json

import pandas as pd
import numpy as np

import streamlit as st

import plotly.express as px
import plotly.graph_objects as go

from const import INPUT_DATA_PATH


################################ Title of Page #####################################################
st.set_page_config(page_title="Cone Data Viewer", page_icon="📈", layout="wide")

st.title("Cone Data Viewer")
#####################################################################################################

################### Get test files, select by material, then material id, then test #################
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
test_selection = st.multiselect(
    "Select tests to compare:",
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
        surf_area = test_metadata["Surface Area (m2)"]
        flux = test_metadata["Heat Flux (kW/m2)"]
        df["t * EHF (kJ/m2)"] = df["Time (s)"] * flux
        df["HRRPUA (kW/m2)"] = df["HRR (kW)"]/ surf_area
        df['dt'] = df["Time (s)"].diff()
        df['Q (MJ)'] = (df['HRR (kW)']*df['dt'])/1000
        df['THR(MJ)'] = df["Q (MJ)"].cumsum()
        df['THRPUA (MJ/m2)'] = df["THR(MJ)"]/surf_area
        if "Mass (g)" in df.columns:
            df["MassPUA (g/m2)"] = df["Mass (g)"]/surf_area
            df['MLR (g/s)'] = abs(np.gradient(df["Mass (g)"])) # COME BACK TO THIS
        else:
            df["Mass (g)"] = None
            df["MassPUA (g/m2)"] = None
        df['MLRPUA (g/s-m2)'] = df['MLR (g/s)']/surf_area
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
        options= ['MassPUA (g/m2)',"MLRPUA (g/s-m2)",'HRRPUA (kW/m2)', "THRPUA (MJ/m2)"]
    )
    else:     
        # Select which column(s) to graph
        y_axis_columns = st.multiselect(
            "Select data to graph across tests",
            options= ['Mass (g)',"MLR (g/s)",'HRR (kW)',"THR (MJ)"]
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