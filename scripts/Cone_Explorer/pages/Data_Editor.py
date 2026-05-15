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
sys.path.append(str(PROJECT_ROOT))
from Cone_Explorer.const import (
    PREPARED_DATA_PATH,
    PREPARED_METADATA_PATH,
    SCRIPT_DIR
)

def safe_savgol_filter(
    data: np.ndarray,
    window_length: int,
    polyorder: int,
    deriv: int,
    delta: float,
) -> np.ndarray:
    """
    Apply Savitzky-Golay filter handling NaN values.

    - If NaNs are only at the beginning or end, filter the valid chunk
    - If NaNs are in the middle, interpolate, filter, then restore NaNs
    """
    nan_mask = np.isnan(data)

    if not nan_mask.any():
        return savgol_filter(
            data,
            window_length=window_length,
            polyorder=polyorder,
            deriv=deriv,
            delta=delta,
        )

    if nan_mask.all():
        return np.full_like(data, np.nan)

    valid_indices = np.where(~nan_mask)[0]
    first_valid = valid_indices[0]
    last_valid = valid_indices[-1]

    middle_section = data[first_valid : last_valid + 1]
    middle_nan_mask = np.isnan(middle_section)

    if not middle_nan_mask.any():
        result = np.full_like(data, np.nan)
        if len(middle_section) >= window_length:
            filtered_middle = savgol_filter(
                middle_section,
                window_length=window_length,
                polyorder=polyorder,
                deriv=deriv,
                delta=delta,
            )
            result[first_valid : last_valid + 1] = filtered_middle
        else:
            result[first_valid : last_valid + 1] = np.gradient(middle_section, delta)
        return result
    else:
        result = np.full_like(data, np.nan)
        data_interp = data.copy()
        valid_mask = ~nan_mask
        x_valid = np.where(valid_mask)[0]
        y_valid = data[valid_mask]
        x_interp = np.arange(first_valid, last_valid + 1)
        data_interp[first_valid : last_valid + 1] = np.interp(
            x_interp, x_valid, y_valid
        )
        middle_interp = data_interp[first_valid : last_valid + 1]

        if len(middle_interp) >= window_length:
            filtered_middle = savgol_filter(
                middle_interp,
                window_length=window_length,
                polyorder=polyorder,
                deriv=deriv,
                delta=delta,
            )
        else:
            filtered_middle = np.gradient(middle_interp, delta)

        result[first_valid : last_valid + 1] = filtered_middle
        result[nan_mask] = np.nan
        return result

################################ Title of Page #####################################################
st.set_page_config(page_title="Cone Data Editor", page_icon="📈", layout="wide")
st.title("Cone Data Editor")

#####################################################################################################
############################## Get test files, select by material, then material id, then test #################
metadata_name_map = {p.stem: p for p in list(PREPARED_METADATA_PATH.rglob("*.json"))}
test_name_map = {p.stem: p for p in list(PREPARED_DATA_PATH.rglob("*.csv"))}

# Date Filtering
if st.checkbox("Filter Tests By Date"):
    st.write("Select the Date Range of Tests You Would Like to View")
    # Initialize session_state variables if they don't already exist
    if "start_date" not in st.session_state:
        st.session_state.start_date = "1982-01-01"  # Default date as string
    if "end_date" not in st.session_state:
        st.session_state.end_date = datetime.today().strftime(
            "%Y-%m-%d"
        )  # Current date as string
    # Text input fields for date entries
    st.session_state.start_date = st.text_input(
        "Start Date (YYYY-MM-DD)", value=st.session_state.start_date
    )
    st.session_state.end_date = st.text_input(
        "End Date (YYYY-MM-DD)", value=st.session_state.end_date
    )

    # Validate date format
    try:
        formatted_start_date = datetime.strptime(
            st.session_state.start_date, "%Y-%m-%d"
        ).strftime("%Y-%m-%d")
        formatted_end_date = datetime.strptime(
            st.session_state.end_date, "%Y-%m-%d"
        ).strftime("%Y-%m-%d")
    except ValueError:
        st.error("Please enter valid dates in YYYY-MM-DD format.")
        st.stop()
    # Modify the test map to only include tests within the date range
    inrange_tests = []
    for test_name, test_path in metadata_name_map.items():
        with open(test_path, "r") as f:
            metadata = json.load(f)
        if formatted_start_date <= metadata["Test Date"] <= formatted_end_date:
            inrange_tests.append(test_name)
    test_name_map = {
        test: test_name_map[test] for test in inrange_tests if test in test_name_map
    }

test_materials = {}  # Generate a list of base materials
test_material_ids = {p: p.split("_")[0] for p in test_name_map}
for matid in test_material_ids.values():
    if "-" in matid:
        test_materials[matid] = matid.split("-")[0]
    else:
        test_materials[matid] = matid

# Select material before selecting specific test
material_selection = st.selectbox(
    "Select a material to view:",
    options=sorted(
        set(test_materials.values())
    ),  # make it a set to only keep the unique values
)

if material_selection:
    id_options = []
    for t in test_materials:
        if test_materials[t] == material_selection:
            id_options += [t]
    # Select material ID
    matid_selection = st.selectbox(
        "Select a material ID to view:",
        options=id_options,
    )

    if matid_selection:
        # Allow selection of only one test with the specified material, and can't select the average (no metadaa file)
        test_options = []
        for t in test_material_ids:
            if test_material_ids[t] == matid_selection and "Average" not in t:
                test_options += [t]

        test_selection = st.selectbox(
            "Select a test to view and edit:",
            options=test_options,
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
                
                # Get subfolder from Original Testname (e.g., "FTT-White/2019" -> "2019")
                og_form = test_metadata.get("Original Source", "")
                subfolder = og_form.split("/")[1] if "/" in og_form else ""
                
                # Build original data path with subfolder
                original_data = prepared_data.replace("Exp-Data_Prepared-Final", "Exp-Data_Parsed")
                if subfolder:
                    # Insert subfolder before filename
                    original_data = original_data.replace(name, f"{subfolder}/{og_name}")
                else:
                    original_data = original_data.replace(name, og_name)
                
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
            else:
                data["HRR (kW)"] = data["HRRPUA (kW/m2)"] * surf_area if surf_area is not None else None
            data["QPUA (MJ/m2)"] = (data['HRRPUA (kW/m2)'] * data['dt']) / 1000
            data["THRPUA (MJ/m2)"] = data["QPUA (MJ/m2)"].cumsum()
            data['Q (MJ)'] = (data['HRR (kW)'] * data['dt']) / 1000
            data['THR (MJ)'] = data["Q (MJ)"].cumsum()

            # Mass and Mass Loss Rate Data
            if "MLR (g/s)" in data.columns:
                data["MLRPUA (g/s-m2)"] = data["MLR (g/s)"] / surf_area if surf_area is not None else None
                data["MassPUA (g/m2)"] = None
                data['Mass Loss (g)'] = None
                data['Mass LossPUA (g/m2)'] = None
            elif "MLRPUA (g/s-m2)" in data.columns:
                data["MLR (g/s)"] = data["MLRPUA (g/s-m2)"] * surf_area if surf_area is not None else None
                data["MassPUA (g/m2)"] = None
                data['Mass Loss (g)'] = None
                data['Mass LossPUA (g/m2)'] = None
            elif "Mass Loss (g)" in data.columns:
                data['Mass (g)'] = (mass - data["Mass Loss (g)"]) if mass is not None else None
                data['MLR (g/s)'] = None
                data['MLRPUA (g/s-m2)'] = None
                data["MassPUA (g/m2)"] = data['Mass (g)'] / surf_area if surf_area is not None else None
            elif "Mass LossPUA (g/m2)" in data.columns:
                data["MassPUA (g/m2)"] = mass - (data["Mass LossPUA (g/m2)"]) if mass is not None else None
                data['Mass (g)'] = data["MassPUA (g/m2)"] * surf_area if surf_area is not None else None
                data['Mass Loss (g)'] = data["Mass LossPUA (g/m2)"] * surf_area if surf_area is not None else None
                data['MLR (g/s)'] = None
                data['MLRPUA (g/s-m2)'] = None
            
            if not data["Mass (g)"].isnull().all():
                dm_dt = safe_savgol_filter(
                            data["Mass (g)"].values,
                            window_length=int(0.08*len(data)) if len(data) >= 50 else 3,  # Adjust window length based on data size, minimum of 3
                            polyorder=2,
                            deriv=1,
                            delta=data['Time (s)'].diff().median(),
                        )# Parameters for savgol filter based on Staggs paper
                data['MLR (g/s)'] = (-1)*dm_dt
                data["MLRPUA (g/s-m2)"] = data["MLR (g/s)"] / surf_area if surf_area is not None else None 
                data["MassPUA (g/m2)"] = data["Mass (g)"]  / surf_area if surf_area is not None else None
                data['Mass Loss (g)'] = mass - data["Mass (g)"] if mass is not None else None
                data['Mass LossPUA (g/m2)'] = (mass / surf_area) - data["MassPUA (g/m2)"] if mass is not None and surf_area is not None else None
            elif not data['MassPUA (g/m2)'].isnull().all():
                data['MLRPUA (g/s-m2)'] = data['MLR (g/s)'] = savgol_filter((-1)*np.gradient(data['MassPUA (g/m2)'],data['Time (s)']),53,3)
                data["MLR (g/s)"] = data["MLRPUA (g/s-m2)"] * surf_area if surf_area is not None else None 
                data["Mass (g)"] = data["MassPUA (g/m2)"] * surf_area if surf_area is not None else None
                data['Mass Loss (g)'] = mass - data["Mass (g)"] if mass is not None else None
                data['Mass LossPUA (g/m2)'] = (mass / surf_area) - data["MassPUA (g/m2)"] if mass is not None and surf_area is not None else None


            #weight air taken from 2077, this publication also used ambient pressure in the building, so will I
            W_air = 28.97
            if amb_pressure:
                data['Rho_Air (kg/m3)'] = ((amb_pressure/1000) * W_air)  / (8.314 * data['T Duct (K)'])
            else:
                data['Rho_Air (kg/m3)'] = None
            data["V Duct (m3/s)"] = data['MFR (kg/s)'] / data["Rho_Air (kg/m3)"]
            
            if 'Extinction Area (m2/kg)' not in data.columns:
                data["Extinction Area (m2/kg)"] = np.divide(
                    (data['V Duct (m3/s)'].astype(float) * data['K Smoke (1/m)'].astype(float)).values,
                    (data['MLR (g/s)'].astype(float) / 1000).values,
                    out=np.zeros(data.shape[0], dtype=float),
                    where=(data['MLR (g/s)'].astype(float).values != 0)
                )
            #Finding Soot  production based on FCD User Guide- but bring area into eq so have Vduct
            #Says to use smoke production sigmas = 8.7m2/g, not sigmaf
            data['Smoke Production (m2/s)'] = data["K Smoke (1/m)"] * data['V Duct (m3/s)']
            data['Smoke ProductionPUA ((m2/s)/m2)'] = data['Smoke Production (m2/s)'] / surf_area if surf_area is not None else None
            data['Soot Production (g/s)'] = 1/8.7 * data['Smoke Production (m2/s)']
            data['Soot ProductionPUA (g/s-m2)'] = data['Soot Production (g/s)'] / surf_area if surf_area is not None else None
            data["HoC (MJ/kg)"] = data["HRRPUA (kW/m2)"] / data["MLRPUA (g/s-m2)"]


            ## Gas Production and Yield
            W_CO2 = 44.01
            W_CO = 28.01
            W_O2 = 32
            if X_O2_i: #FTT Data and Hopefully other good data (ie Netzch)
                #Production and yields calculated by following FCD user guide
                data['CO2 Production (g/s)'] = (W_CO2/W_air) * (data['CO2 (Vol fr)'] - X_CO2_i) * data['MFR (kg/s)'] *1000
                data['CO Production (g/s)'] = (W_CO/W_air) * (data['CO (Vol fr)'] - X_CO_i) * data['MFR (kg/s)'] *1000
                data['O2 Consumption (g/s)'] = (W_O2/W_air) * (X_O2_i - data['O2 (Vol fr)'] ) * data['MFR (kg/s)'] *1000
                data['CO ProductionPUA (g/s-m2)'] = data['CO Production (g/s)'] / surf_area if surf_area is not None else None
                data['CO2 ProductionPUA (g/s-m2)'] = data['CO2 Production (g/s)'] / surf_area if surf_area is not None else None
                data['O2 ConsumptionPUA (g/s-m2)'] = data['O2 Consumption (g/s)'] / surf_area if surf_area is not None else None
                
                #Not using right now, but keep the code if we ever want it
                data['O2'] = (data['O2 Consumption (g/s)'] * data['dt'])
                data['Total O2'] = data['O2'].cumsum()
                # For CO
                data['CO'] = data['CO Production (g/s)'] * data['dt']
                data['Total CO'] = data['CO'].cumsum()

                # For CO2
                data['CO2'] = data['CO2 Production (g/s)'] * data['dt']
                data['Total CO2'] = data['CO2'].cumsum()

                # Yields
                data['CO2 (kg/kg)'] = np.divide(
                    data['CO2 Production (g/s)'].astype(float).values,
                    data['MLR (g/s)'].astype(float).values,
                    out=np.zeros(data.shape[0], dtype=float),
                    where=(data['MLR (g/s)'].astype(float).values != 0)
                )
                data['CO (kg/kg)'] = np.divide(
                    data['CO Production (g/s)'].astype(float).values,
                    data['MLR (g/s)'].astype(float).values,
                    out=np.zeros(data.shape[0], dtype=float),
                    where=(data['MLR (g/s)'].astype(float).values != 0)
                )


            else:
                data['CO2 Production (g/s)'] = None
                data['CO Production (g/s)'] = None
                data['O2 Consumption (g/s)'] = None
                data['CO2 ProductionPUA (g/s-m2)'] = None
                data['CO ProductionPUA (g/s-m2)'] = None
                data['O2 ConsumptionPUA (g/s-m2)'] = None

            # For Soot
            data['Soot'] = data['Soot Production (g/s)'] * data['dt']
            data['Total Soot'] = data['Soot'].cumsum()

            data['Smoke'] = data['Smoke Production (m2/s)'] * data['dt']
            data['Total Smoke'] = data['Smoke'].cumsum()


            #Remaining gasses if they don't exist
            for gas in ["CO2 (kg/kg)", "CO (kg/kg)","H2O (kg/kg)", "HCl (kg/kg)", "H'carbs (kg/kg)"]:
                if gas not in data.columns:
                    data[gas] = None
            test_data = data.copy()

        ######################################################################################################################################################################

        ############################################### Generate Plot #########################################################                 
            normalize = st.checkbox("Normalize Y-Axis Data by Surface Area (m2)")
            additional = st.checkbox("View Additional Calculated Properties")
            if not normalize and not additional:
                options = [
                    'HRR (kW)','Mass (g)', "MLR (g/s)",  "THR (MJ)", 
                    "O2 (Vol fr)", "CO2 (Vol fr)", "CO (Vol fr)", "K Smoke (1/m)"
                    
                ]
                default_value = 'HRR (kW)'
                default_index = options.index(default_value) if default_value in options else 0
                columns_to_graph = st.selectbox(
                    "Select data to graph across tests",
                    options=options,
                    index=default_index
                )
            elif normalize and not additional:
                options = [
                    'HRRPUA (kW/m2)','MassPUA (g/m2)', "MLRPUA (g/s-m2)", "THRPUA (MJ/m2)", 
                "O2 (Vol fr)", "CO2 (Vol fr)", "CO (Vol fr)", "K Smoke (1/m)"
                ] 
                default_value = 'HRRPUA (kW/m2)'
                default_index = options.index(default_value) if default_value in options else 0
                columns_to_graph = st.selectbox(
                    "Select data to graph across tests",
                    options=options,
                    index=default_index
                )
            elif not normalize and additional:
                options = [
                    'Mass Loss (g)',
                "CO2 Production (g/s)", 'CO2 (kg/kg)', "CO Production (g/s)", "CO (kg/kg)", "O2 Consumption (g/s)", "H2O (kg/kg)",
                "HCl (kg/kg)", "H'carbs (kg/kg)",
                "Soot Production (g/s)", 'Smoke Production (m2/s)', "Extinction Area (m2/kg)","MFR (kg/s)",'V Duct (m3/s)'
                ]
                default_value = 'Mass Loss (g)'
                default_index = options.index(default_value) if default_value in options else 0
                columns_to_graph = st.selectbox(
                    "Select data to graph across tests",
                    options=options,
                    index=default_index
                )
            else:
                options = [
                    'Mass LossPUA (g/m2)',
                "CO2 ProductionPUA (g/s-m2)", 'CO2 (kg/kg)', "CO ProductionPUA (g/s-m2)", "CO (kg/kg)", "O2 ConsumptionPUA (g/s-m2)", "H2O (kg/kg)",
                "HCl (kg/kg)", "H'carbs (kg/kg)",
                "Soot ProductionPUA (g/s-m2)", 'Smoke ProductionPUA ((m2/s)/m2)', "Extinction Area (m2/kg)","MFR (kg/s)",'V Duct (m3/s)'
                ]
                default_value = 'Mass LossPUA (g/m2)'
                default_index = options.index(default_value) if default_value in options else 0
                columns_to_graph = st.selectbox(
                    "Select data to graph across tests",
                    options=options,
                    index=default_index
                )
            st.caption('Legacy Data May Be Missing Several Data Columns')
            if not test_data[columns_to_graph].isnull().all():
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
                    if column != 'Time (s)':
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
                
                # Export full dataframe button
                st.markdown("---")
                st.subheader("Export Data")
                
                # Create a download button for the full dataframe
                csv = test_data.to_csv(index=False)
                st.download_button(
                    label="Download Full Test Data as CSV",
                    data=csv,
                    file_name=f"{test_selection}_full_data.csv",
                    mime="text/csv",
                )
            ################################################ Saving Adjusted/ Clipped Data ########################################################################
                # Save the adjusted data to a specified path
                st.sidebar.markdown("This button only saves data clipping, csv file modifications are saved seperatley")
                if st.sidebar.button("Save Clipped Data"):
                    save_path = str(test_name_map[test_selection])
                    save_dir = Path(save_path).parent
                    save_dir.mkdir(parents=True, exist_ok=True)
                    columns = pd.read_csv(test_name_map[test_selection], nrows=0).columns.tolist()
                    data_out = data_copy[columns].copy()
                    data_out.dropna(how='all', inplace=True)
                    data_out.to_csv(
                        save_path,
                        index=False,
                    )

                    st.sidebar.success(f"Data saved to {save_path}.")
                    #adjust metadata
                    with open(metadata_name_map[test_selection], 'r') as f:
                        metadata = json.load(f)
                    if cutoff_start < 0:
                        lowerbound = 0
                    else:
                        lowerbound = cutoff_start
                    if cutoff_end > data_copy['Time (s)'].max():
                        upperbound = data_copy['Time (s)'].max()
                    else:
                        upperbound = cutoff_end
                    test_metadata["Data Corrections"].append(f"{date}: Data from {lowerbound}s to {upperbound}s was removed")
                    test_metadata['Manually Prepared'] = date
                    with open(metadata_name_map[test_selection], "w") as f:
                        json.dump(test_metadata, f, indent=4)
            
            
            else:
                st.caption(f"⚠️ Warning: **{test_selection}** does not contain **{columns_to_graph}** data.")
        ########################################################################################################################################
 
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
