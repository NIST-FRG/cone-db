"""C-Factor Calibration Calculator based on ASTM E1354."""

import sys

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from math import sqrt
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]  # .../Scripts
sys.path.append(str(PROJECT_ROOT))
from Cone_Explorer.const import (
    SCRIPT_DIR
)


st.set_page_config(page_title="C-Factor Calculator", page_icon="🔥", layout="wide")

st.title("🔥 C-Factor Calibration Calculator")
st.markdown("Based on **ASTM E1354 Appendix A1.4**")


# --- Constants ---
DELTA_HC_METHANE = 50.01  # MJ/kg - Heat of combustion of methane, from user guide to cone and nbsir82-2602
E_DEFAULT = 12.54  # MJ/kg - ΔHc/r_0 for methane


# --- Helper Functions ---

def calculate_X_H2O(amb_temp_C, rel_humid_pct, amb_pressure_Pa):
    """Calculate water vapor mole fraction from ambient conditions."""
    p_sat_water = 6.1078 * 10**((7.5 * amb_temp_C) / (237.3 + amb_temp_C)) * 100
    p_h2o = (rel_humid_pct / 100) * p_sat_water
    X_H2O = p_h2o / amb_pressure_Pa
    return X_H2O


def slpm_to_kg_s_methane(slpm):
    """Convert methane flow from SLPM to kg/s."""
    M_CH4 = 16.04e-3  # kg/mol
    V_molar_STP = 22.414  # L/mol at STP
    mol_per_min = slpm / V_molar_STP
    kg_per_s = (mol_per_min * M_CH4) / 60
    return kg_per_s


def calculate_hrr_methane(m_dot_kg_s):
    """Calculate HRR from methane mass flow rate. Returns MW."""
    return m_dot_kg_s * DELTA_HC_METHANE


def calculate_odf(X_O2, X_CO2, X_CO, X_O2_0, X_CO2_0):
    """Calculate oxygen depletion factor (phi) per ASTM E1354 A1.4."""
    numerator = X_O2_0 * (1 - X_CO2 - X_CO) - X_O2 * (1 - X_CO2_0)
    denominator = X_O2_0 * (1 - X_CO2 - X_CO - X_O2)
    if denominator == 0:
        return None
    return numerator / denominator


def calculate_c_factor(q_dot_MW, E, X_O2_amb, delta_P, T_e, phi, X_CO, X_O2):
    """Calculate C-factor by rearranging ASTM E1354 A1.4 HRR equation."""
    if delta_P <= 0 or T_e <= 0 or phi is None:
        return None
    
    if X_O2 > 0:
        co_correction = 0.172 * (1 - phi) * (X_CO / X_O2)
    else:
        co_correction = 0
    
    bracket_term = (phi - co_correction) / (1 - phi + 1.105 * phi)
    if bracket_term == 0:
        return None
    
    flow_term = sqrt(delta_P / T_e)
    denominator = 1.10 * E * X_O2_amb * flow_term * bracket_term
    if denominator == 0:
        return None
    
    return q_dot_MW / denominator


def calculate_c_factor_for_row(row, X_O2_0, X_CO2_0, X_O2_amb, E, methane_col):
    """Calculate C-factor for a single row of data."""
    try:
        X_O2 = row['O2 (%)'] / 100 if pd.notna(row.get('O2 (%)')) else None
        X_CO2 = row['CO2 (%)'] / 100 if pd.notna(row.get('CO2 (%)')) else None
        X_CO = row['CO (%)'] / 100 if pd.notna(row.get('CO (%)')) else None
        delta_P = row['DPT (Pa)'] if pd.notna(row.get('DPT (Pa)')) else None
        T_e = row['Stack TC (K)'] if pd.notna(row.get('Stack TC (K)')) else None
        methane_slpm = row[methane_col] if methane_col and pd.notna(row.get(methane_col)) else None
        
        if any(v is None for v in [X_O2, X_CO2, X_CO, delta_P, T_e, methane_slpm]):
            return np.nan
        if delta_P <= 0 or T_e <= 0:
            return np.nan
        
        m_dot = slpm_to_kg_s_methane(methane_slpm)
        q_dot_MW = calculate_hrr_methane(m_dot)
        phi = calculate_odf(X_O2, X_CO2, X_CO, X_O2_0, X_CO2_0)
        
        if phi is None or phi <= 0:
            return np.nan
        
        c = calculate_c_factor(q_dot_MW, E, X_O2_amb, delta_P, T_e, phi, X_CO, X_O2)
        return c if c is not None else np.nan
    except Exception:
        return np.nan


def parse_ftt_file(file_path=None, file_content=None, encoding='cp1252'):
    """Parse FTT cone calorimeter CSV file."""
    if file_path is not None:
        df = pd.read_csv(file_path, encoding=encoding)
    elif file_content is not None:
        from io import StringIO
        df = pd.read_csv(StringIO(file_content))
    else:
        raise ValueError("Must provide either file_path or file_content")
    
    if len(df) > 1 and df.iloc[1].isnull().all():
        df = df.drop(1)
    df = df.dropna(how="all")
    
    raw_metadata = df[df.columns[:2]].dropna(how="all")
    raw_metadata = raw_metadata.T
    new_header = raw_metadata.iloc[0]
    raw_metadata = raw_metadata[1:]
    raw_metadata.columns = new_header
    raw_metadata = raw_metadata.to_dict(orient="list")
    metadata = {k: v[0] if len(v) > 0 else None for k, v in raw_metadata.items()}
    
    data = df[df.columns[2:]].copy()
    data = data.reset_index(drop=True)
    for col in data.columns:
        data[col] = pd.to_numeric(data[col], errors='coerce')
    time_col = data.columns[0]
    data = data.dropna(subset=[time_col])
    
    return metadata, data


def get_number(metadata, key, default=None):
    """Safely extract a numeric value from metadata."""
    try:
        val = metadata.get(key)
        if val is None:
            return default
        return float(val)
    except (ValueError, TypeError):
        return default


def get_string(metadata, key, default="N/A"):
    """Safely extract a string value from metadata."""
    val = metadata.get(key)
    return str(val) if val is not None else default


def reset_adjustments():
    """Reset all adjustment values to defaults."""
    keys_to_reset = [k for k in st.session_state.keys() 
                     if k.startswith(('scale_', 'shift_', 'param_', 'region_'))]
    for key in keys_to_reset:
        del st.session_state[key]


# --- Main App ---

st.header("📁 Load Calibration Data")

uploaded_file = st.file_uploader(
    "Upload calibration file (.CSV)",
    type=['csv', 'CSV'],
    help="Upload the FTT cone calorimeter calibration data file"
)

st.markdown("**Or** enter file path directly:")
file_path = st.text_input(
    "File path",
    placeholder=r"C:\CC5\CALIB\C2410010.CSV",
    help="Full path to the calibration file"
)

data_loaded = False
metadata = {}
data = None

if 'current_file' not in st.session_state:
    st.session_state.current_file = None

# Then in the file loading sections:
if uploaded_file is not None:
    try:
        file_bytes = uploaded_file.read()
        file_content = file_bytes.decode('cp1252')
        metadata, data = parse_ftt_file(file_content=file_content)
        data_loaded = True
        
        # Only clear if file changed
        if st.session_state.current_file != uploaded_file.name:
            st.session_state.current_file = uploaded_file.name
            st.cache_data.clear()
            reset_adjustments()
        shortname = Path(uploaded_file.name).name
        st.success(f"✅ Loaded: {shortname} ({len(data)} data points)")
    except Exception as e:
        st.error(f"Error reading file: {e}")
        import traceback
        st.code(traceback.format_exc())

elif file_path:
    try:
        metadata, data = parse_ftt_file(file_path=file_path, encoding='cp1252')
        data_loaded = True
        shortname = Path(file_path).name
        st.success(f"✅ Loaded: {shortname} ({len(data)} data points)")

        # Only clear if file changed
        if st.session_state.current_file != file_path:
            st.session_state.current_file = file_path
            st.cache_data.clear()
            reset_adjustments()
        
        st.success(f"✅ Loaded: {shortname} ({len(data)} data points)")
    except FileNotFoundError:
        st.error(f"File not found: {file_path}")
    except Exception as e:
        st.error(f"Error reading file: {e}")
        import traceback
        st.code(traceback.format_exc())


if data_loaded and data is not None and len(data) > 0:
    
    # --- Prepare data and extract defaults ---
    time_col = data.columns[0]
    if time_col != 'Time (s)':
        data = data.rename(columns={time_col: 'Time (s)'})
    
    methane_col = None
    for col in data.columns:
        if 'methane' in col.lower() and 'mfm' in col.lower():
            methane_col = col
            break
    if not methane_col:
        for col in data.columns:
            if 'ch4' in col.lower() or 'methane' in col.lower():
                methane_col = col
                break
    
    # Get burner timing from metadata (fixed, not adjustable)
    burner_on_s = int(get_number(metadata, 'Burner on (s)', 65) or 65)
    burner_off_s = int(get_number(metadata, 'Burner off (s)', 425) or 425)
    
    # Get default values from metadata
    default_amb_temp = get_number(metadata, 'Ambient temperature (°C)', 20.0)
    default_amb_press = get_number(metadata, 'Barometric pressure (Pa)', 101325.0)
    default_rel_humid = get_number(metadata, 'Relative humidity (%)', 50.0)
    default_o2_delay = int(get_number(metadata, 'O2 delay time (s)', 0) or 0)
    default_co2_delay = int(get_number(metadata, 'CO2 delay time (s)', 0) or 0)
    default_co_delay = int(get_number(metadata, 'CO delay time (s)', 0) or 0)
    
    time_max = int(data['Time (s)'].max())
    
    # --- SIDEBAR: All Adjustments ---
    with st.sidebar:
        st.header("🔧 Adjustments")
        
        if st.button("🔄 Reset to Defaults", type="secondary", use_container_width=True):
            reset_adjustments()
            st.rerun()
        
        st.caption("Modify parameters to see how they affect the calculated C-factor.")
        
        # Track all adjustments
        adjustments = {
            'parameters': {},
            'columns': {}
        }
        
        # --- Steady State Region Selection ---
        st.subheader("📊 Steady State Regions")
        
        st.markdown("**Baseline Period**")
        baseline_start = st.number_input(
            "Start (s)", min_value=0, max_value=time_max, value=10, step=1,
            key="region_baseline_start"
        )
        baseline_end = st.number_input(
            "End (s)", min_value=0, max_value=time_max,
            value=min(burner_on_s - 5, time_max), step=1,
            key="region_baseline_end"
        )
        
        st.markdown("**Calibration Period**")
        default_cal_start = burner_on_s + 240
        default_cal_end = burner_off_s - 5
        cal_start = st.number_input(
            "Start (s)", min_value=0, max_value=time_max,
            value=default_cal_start, step=1,
            key="region_cal_start"
        )
        cal_end = st.number_input(
            "End (s)", min_value=0, max_value=time_max,
            value=min(default_cal_end, time_max), step=1,
            key="region_cal_end"
        )
        
        st.divider()
        
        # --- Parameter Adjustments (Static Values) ---
        st.subheader("📌 Static Parameters")
        
        st.markdown("**Ambient Conditions**")
        
        amb_temp_C = st.number_input(
            "Ambient Temp (°C)", min_value=-20.0, max_value=50.0,
            value=float(default_amb_temp) if default_amb_temp else 20.0, step=0.1,
            help=f"File value: {default_amb_temp}",
            key="param_amb_temp"
        )
        amb_pressure_Pa = st.number_input(
            "Pressure (Pa)", min_value=80000.0, max_value=110000.0,
            value=float(default_amb_press) if default_amb_press else 101325.0, step=100.0,
            help=f"File value: {default_amb_press}",
            key="param_amb_press"
        )
        rel_humid_pct = st.number_input(
            "Rel. Humidity (%)", min_value=0.0, max_value=100.0,
            value=float(default_rel_humid) if default_rel_humid else 50.0, step=0.1,
            help=f"File value: {default_rel_humid}",
            key="param_rel_humid"
        )
        
        # Track changes
        if default_amb_temp and amb_temp_C != default_amb_temp:
            adjustments['parameters']['Ambient Temp (°C)'] = {'file': default_amb_temp, 'current': amb_temp_C}
        if default_amb_press and amb_pressure_Pa != default_amb_press:
            adjustments['parameters']['Pressure (Pa)'] = {'file': default_amb_press, 'current': amb_pressure_Pa}
        if default_rel_humid and rel_humid_pct != default_rel_humid:
            adjustments['parameters']['Rel. Humidity (%)'] = {'file': default_rel_humid, 'current': rel_humid_pct}
        
        st.divider()
        
        st.markdown("**Calculation Constants**")
        
        E_value = st.number_input(
            "E = ΔHc/r₀ (MJ/kg)", min_value=10.0, max_value=15.0,
            value=E_DEFAULT, step=0.01,
            help=f"Default: {E_DEFAULT} MJ/kg (methane)",
            key="param_E"
        )
        
        st.caption(f"ΔHc methane: {DELTA_HC_METHANE} MJ/kg")
        
        if E_value != E_DEFAULT:
            adjustments['parameters']['E (MJ/kg)'] = {'file': E_DEFAULT, 'current': E_value}
        
        st.divider()
        
        st.markdown("**Analyzer Delay Times**")
        
        o2_delay = st.number_input(
            "O₂ Delay (s)", min_value=0, max_value=60,
            value=default_o2_delay, step=1,
            help=f"File value: {default_o2_delay}",
            key="param_o2_delay"
        )
        co2_delay = st.number_input(
            "CO₂ Delay (s)", min_value=0, max_value=60,
            value=default_co2_delay, step=1,
            help=f"File value: {default_co2_delay}",
            key="param_co2_delay"
        )
        co_delay = st.number_input(
            "CO Delay (s)", min_value=0, max_value=60,
            value=default_co_delay, step=1,
            help=f"File value: {default_co_delay}",
            key="param_co_delay"
        )
        
        if o2_delay != default_o2_delay:
            adjustments['parameters']['O2 Delay (s)'] = {'file': default_o2_delay, 'current': o2_delay}
        if co2_delay != default_co2_delay:
            adjustments['parameters']['CO2 Delay (s)'] = {'file': default_co2_delay, 'current': co2_delay}
        if co_delay != default_co_delay:
            adjustments['parameters']['CO Delay (s)'] = {'file': default_co_delay, 'current': co_delay}
        
        st.divider()
        
        # --- Column Adjustments (Scale & Shift) ---
        st.subheader("📊 Column Adjustments")
        st.caption("Value = (Original × Scale) + Shift")
        
        column_configs = {
            'O2 (%)': {'scale_step': 0.001, 'shift_step': 0.01, 'shift_range': (-1.0, 1.0)},
            'CO2 (%)': {'scale_step': 0.001, 'shift_step': 0.001, 'shift_range': (-0.1, 0.1)},
            'CO (%)': {'scale_step': 0.001, 'shift_step': 0.0001, 'shift_range': (-0.01, 0.01)},
            'DPT (Pa)': {'scale_step': 0.001, 'shift_step': 0.1, 'shift_range': (-10.0, 10.0)},
            'Stack TC (K)': {'scale_step': 0.001, 'shift_step': 0.1, 'shift_range': (-10.0, 10.0)},
        }
        
        if methane_col:
            column_configs[methane_col] = {'scale_step': 0.001, 'shift_step': 0.01, 'shift_range': (-1.0, 1.0)}
        
        col_keys = [k for k in column_configs.keys() if k in data.columns]
        
        for col_key in col_keys:
            cfg = column_configs[col_key]
            
            with st.expander(f"**{col_key}**"):
                scale = st.number_input(
                    "Scale", min_value=0.5, max_value=2.0,
                    value=1.0, step=cfg['scale_step'],
                    key=f"scale_{col_key}", format="%.4f"
                )
                shift = st.number_input(
                    "Shift", min_value=cfg['shift_range'][0], max_value=cfg['shift_range'][1],
                    value=0.0, step=cfg['shift_step'],
                    key=f"shift_{col_key}", format="%.4f"
                )
                
                if scale != 1.0 or shift != 0.0:
                    adjustments['columns'][col_key] = {'scale': scale, 'shift': shift}
        
        # Check if any adjustments are active
        any_adjustments = len(adjustments['parameters']) > 0 or len(adjustments['columns']) > 0
        
        if any_adjustments:
            st.divider()
            st.warning("⚠️ **Adjustments Active**")
            
            if adjustments['parameters']:
                st.markdown("**Parameters:**")
                for param, vals in adjustments['parameters'].items():
                    st.caption(f"{param}: {vals['file']} → {vals['current']}")
            
            if adjustments['columns']:
                st.markdown("**Columns:**")
                for col, adj in adjustments['columns'].items():
                    st.caption(f"{col}: ×{adj['scale']:.4f}, +{adj['shift']:.4f}")
    
    # --- Calculate derived values (after sidebar, before main content) ---
    X_H2O_initial = calculate_X_H2O(amb_temp_C, rel_humid_pct, amb_pressure_Pa)
    
    # --- Apply Adjustments to Data ---
    data_adjusted = data.copy()
    
    # Apply column scale and shift
    for col_key in col_keys:
        if col_key in adjustments['columns']:
            adj = adjustments['columns'][col_key]
            data_adjusted[col_key] = (data_adjusted[col_key] * adj['scale']) + adj['shift']
    
    # Calculate HRR from Methane
    if methane_col:
        data_adjusted['m_dot_CH4 (kg/s)'] = data_adjusted[methane_col].apply(
            lambda x: slpm_to_kg_s_methane(x) if pd.notna(x) else np.nan
        )
        data_adjusted['HRR_CH4 (MW)'] = data_adjusted['m_dot_CH4 (kg/s)'].apply(
            lambda x: calculate_hrr_methane(x) if pd.notna(x) else np.nan
        )
        data_adjusted['HRR_CH4 (kW)'] = data_adjusted['HRR_CH4 (MW)'] * 1000
    
    # Apply delay time corrections
    data_corrected = data_adjusted.copy()
    
    if o2_delay > 0 and 'O2 (%)' in data_corrected.columns:
        data_corrected['O2 (%)'] = data_corrected['O2 (%)'].shift(-o2_delay)
    if co2_delay > 0 and 'CO2 (%)' in data_corrected.columns:
        data_corrected['CO2 (%)'] = data_corrected['CO2 (%)'].shift(-co2_delay)
    if co_delay > 0 and 'CO (%)' in data_corrected.columns:
        data_corrected['CO (%)'] = data_corrected['CO (%)'].shift(-co_delay)
    
    max_delay = max(o2_delay, co2_delay, co_delay)
    if max_delay > 0:
        data_corrected = data_corrected.iloc[:-max_delay]
    
    # Create masks
    baseline_mask = (data_corrected['Time (s)'] >= baseline_start) & (data_corrected['Time (s)'] <= baseline_end)
    calibration_mask = (data_corrected['Time (s)'] >= cal_start) & (data_corrected['Time (s)'] <= cal_end)
    
    n_baseline = baseline_mask.sum()
    n_calibration = calibration_mask.sum()
    
    # Calculate Baseline Averages
    baseline_df = data_corrected[baseline_mask]
    cal_df = data_corrected[calibration_mask]
    
    if len(baseline_df) > 0:
        X_O2_0 = baseline_df['O2 (%)'].mean() / 100 if 'O2 (%)' in baseline_df.columns else 0.2095
        X_CO2_0 = baseline_df['CO2 (%)'].mean() / 100 if 'CO2 (%)' in baseline_df.columns else 0.0004
        X_CO_0 = baseline_df['CO (%)'].mean() / 100 if 'CO (%)' in baseline_df.columns else 0.0
        X_O2_amb = (1 - X_H2O_initial) * X_O2_0
    else:
        X_O2_0 = 0.2095
        X_CO2_0 = 0.0004
        X_CO_0 = 0.0
        X_O2_amb = (1 - X_H2O_initial) * X_O2_0
    
    # Calculate Time-Series C-Factor
    if methane_col:
        data_corrected['C-Factor'] = data_corrected.apply(
            lambda row: calculate_c_factor_for_row(row, X_O2_0, X_CO2_0, X_O2_amb, E_value, methane_col),
            axis=1
        )
        cal_df = data_corrected[calibration_mask]
        baseline_df = data_corrected[baseline_mask]
    
    # ==================== MAIN CONTENT ====================
    
    st.divider()
    
    with st.expander("🔍 Debug: View raw column names"):
        st.write("**Metadata keys:**")
        st.write(list(metadata.keys()))
        st.write("**Data columns:**")
        st.write(list(data.columns))
    
    # --- Display Metadata ---
    st.header("📋 Test Metadata")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**Test Information**")
        st.text(f"Date: {get_string(metadata, 'Date of test')}")
        st.text(f"Time: {get_string(metadata, 'Time of test')}")
        st.text(f"Filename: {get_string(metadata, 'Filename')}")
    
    with col2:
        st.markdown("**Calibration Parameters**")
        st.text(f"HRR level (target): {get_number(metadata, 'HRR level (kW)', 'N/A')} kW")
        st.text(f"Burner on: {burner_on_s} s")
        st.text(f"Burner off: {burner_off_s} s")
    
    with col3:
        st.markdown("**C-Factor Values from File**")
        st.text(f"Previous: {get_number(metadata, 'Initial C-factor (SI units)', 'N/A')}")
        st.text(f"Mean: {get_number(metadata, 'Mean C-factor', 'N/A')}")
        st.text(f"ISO 5660-1: {get_number(metadata, 'ISO 5660-1 C-factor', 'N/A')}")
    
    with st.expander("View all metadata"):
        for key, value in metadata.items():
            st.text(f"{key}: {value}")
    
    st.divider()
    
    # --- Methane column status ---
    if not methane_col:
        st.error("⚠️ Methane MFM column not found!")
    
    # --- Data Preview ---
    st.header("📊 Data Preview")
    
    col1, col2 = st.columns([1, 3])
    
    with col1:
        preview_region = st.radio(
            "Select region:",
            options=["Baseline", "Calibration"],
            horizontal=False
        )
    
    preview_cols = ['Time (s)']
    if methane_col and methane_col in data_corrected.columns:
        preview_cols.append(methane_col)
    for col in ['O2 (%)', 'CO2 (%)', 'CO (%)', 'DPT (Pa)', 'Stack TC (K)', 'HRR_CH4 (kW)', 'C-Factor']:
        if col in data_corrected.columns:
            preview_cols.append(col)
    
    with col2:
        if preview_region == "Baseline":
            preview_df = baseline_df[preview_cols]
            st.caption(f"Baseline: {baseline_start}s - {baseline_end}s ({len(preview_df)} points)")
        else:
            preview_df = cal_df[preview_cols]
            st.caption(f"Calibration: {cal_start}s - {cal_end}s ({len(preview_df)} points)")
        
        if len(preview_df) > 0:
            st.dataframe(preview_df, use_container_width=True, hide_index=True)
        else:
            st.warning("No data points in selected region.")
    
    st.divider()
    
    # --- Plot Time Series Data ---
    st.header("📈 Time Series Data")
    
    available_plots = []
    if 'HRR_CH4 (kW)' in data_corrected.columns:
        available_plots.append(('HRR_CH4 (kW)', 'HRR from Methane (kW)', 'red'))
    if 'O2 (%)' in data_corrected.columns:
        available_plots.append(('O2 (%)', 'Oxygen (%)', 'blue'))
    if 'DPT (Pa)' in data_corrected.columns:
        available_plots.append(('DPT (Pa)', 'Differential Pressure (Pa)', 'brown'))
    if 'C-Factor' in data_corrected.columns:
        available_plots.append(('C-Factor', 'C-Factor (time-resolved)', 'black'))
    
    n_plots = len(available_plots)
    
    if n_plots > 0:
        n_cols = 2
        n_rows = (n_plots + 1) // 2
        
        fig = make_subplots(
            rows=n_rows, cols=n_cols,
            subplot_titles=[p[1] for p in available_plots],
            vertical_spacing=0.12, horizontal_spacing=0.08
        )
        
        for idx, (col_name, title, color) in enumerate(available_plots):
            row = idx // 2 + 1
            col = idx % 2 + 1
            
            fig.add_trace(
                go.Scatter(
                    x=data_corrected['Time (s)'], y=data_corrected[col_name],
                    name=title, line=dict(color=color, width=1)
                ),
                row=row, col=col
            )
            
            fig.add_vrect(x0=baseline_start, x1=baseline_end, fillcolor="lightblue",
                          opacity=0.3, layer="below", line_width=0, row=row, col=col)
            fig.add_vrect(x0=cal_start, x1=cal_end, fillcolor="lightgreen",
                          opacity=0.3, layer="below", line_width=0, row=row, col=col)
            fig.add_vline(x=burner_on_s, line_dash="dash", line_color="green", row=row, col=col)
            fig.add_vline(x=burner_off_s, line_dash="dash", line_color="red", row=row, col=col)
            
            if col_name == 'C-Factor':
                c_factor_cal_data = cal_df['C-Factor'].dropna()
                if len(c_factor_cal_data) > 0:
                    c_min = c_factor_cal_data.min()
                    c_max = c_factor_cal_data.max()
                    c_range = c_max - c_min if c_max != c_min else 0.001
                    fig.update_yaxes(range=[c_min - 0.1 * c_range, c_max + 0.1 * c_range], row=row, col=col)
        
        fig.update_layout(height=300 * n_rows, showlegend=False, title_text="Calibration Data")
        st.plotly_chart(fig, use_container_width=True)
        st.caption("🔵 Blue = Baseline | 🟢 Green = Calibration | Dashed: 🟢 Burner On | 🔴 Burner Off")
    
    st.divider()
    
    # --- Steady State Averages ---
    st.header("📊 Steady State Averages")
    
    if len(baseline_df) == 0 or len(cal_df) == 0:
        st.error("Insufficient data points in selected regions!")
    else:
        T_e_0 = baseline_df['Stack TC (K)'].mean() if 'Stack TC (K)' in baseline_df.columns else 298
        delta_P_0 = baseline_df['DPT (Pa)'].mean() if 'DPT (Pa)' in baseline_df.columns else 0
        
        X_O2_cal = cal_df['O2 (%)'].mean() / 100 if 'O2 (%)' in cal_df.columns else 0.2095
        X_CO2_cal = cal_df['CO2 (%)'].mean() / 100 if 'CO2 (%)' in cal_df.columns else 0.0004
        X_CO_cal = cal_df['CO (%)'].mean() / 100 if 'CO (%)' in cal_df.columns else 0.0
        T_e_cal = cal_df['Stack TC (K)'].mean() if 'Stack TC (K)' in cal_df.columns else 400
        delta_P_cal = cal_df['DPT (Pa)'].mean() if 'DPT (Pa)' in cal_df.columns else 70
        
        if methane_col and methane_col in cal_df.columns:
            methane_slpm_cal = cal_df[methane_col].mean()
            methane_slpm_std = cal_df[methane_col].std()
            m_dot_CH4_kg_s = slpm_to_kg_s_methane(methane_slpm_cal)
            m_dot_CH4_kg_s_std = slpm_to_kg_s_methane(methane_slpm_std)
            q_dot_CH4_MW = calculate_hrr_methane(m_dot_CH4_kg_s)
            q_dot_CH4_MW_std = calculate_hrr_methane(m_dot_CH4_kg_s_std)
            q_dot_CH4_kW = q_dot_CH4_MW * 1000
            q_dot_CH4_kW_std = q_dot_CH4_MW_std * 1000
        else:
            methane_slpm_cal = methane_slpm_std = None
            m_dot_CH4_kg_s = m_dot_CH4_kg_s_std = None
            q_dot_CH4_MW = q_dot_CH4_MW_std = None
            q_dot_CH4_kW = q_dot_CH4_kW_std = None
        
        phi_iso = calculate_odf(X_O2_cal, X_CO2_cal, X_CO_cal, X_O2_0, X_CO2_0)
        target_hrr = get_number(metadata, 'HRR level (kW)')
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader(f"Baseline ({n_baseline} points)")
            baseline_stats = pd.DataFrame({
                'Parameter': ['X_O₂', 'X_CO₂', 'X_CO', 'T_e (K)', 'ΔP (Pa)'],
                'Mean': [
                    f"{X_O2_0:.6f}", 
                    f"{X_CO2_0:.6f}", 
                    f"{X_CO_0:.8f}",
                    f"{T_e_0:.1f}", 
                    f"{delta_P_0:.2f}"
                ],
                'Std Dev': [
                    f"{baseline_df['O2 (%)'].std()/100:.6f}" if 'O2 (%)' in baseline_df.columns else "N/A",
                    f"{baseline_df['CO2 (%)'].std()/100:.6f}" if 'CO2 (%)' in baseline_df.columns else "N/A",
                    f"{baseline_df['CO (%)'].std()/100:.8f}" if 'CO (%)' in baseline_df.columns else "N/A",
                    f"{baseline_df['Stack TC (K)'].std():.1f}" if 'Stack TC (K)' in baseline_df.columns else "N/A",
                    f"{baseline_df['DPT (Pa)'].std():.2f}" if 'DPT (Pa)' in baseline_df.columns else "N/A"
                ]
            })
            st.dataframe(baseline_stats, use_container_width=True, hide_index=True)
        
        with col2:
            st.subheader(f"Calibration ({n_calibration} points)")
            
            # Build calibration stats including methane
            cal_params = ['X_O₂', 'X_CO₂', 'X_CO', 'T_e (K)', 'ΔP (Pa)']
            cal_means = [
                f"{X_O2_cal:.6f}", 
                f"{X_CO2_cal:.6f}", 
                f"{X_CO_cal:.8f}",
                f"{T_e_cal:.1f}", 
                f"{delta_P_cal:.2f}"
            ]
            cal_stds = [
                f"{cal_df['O2 (%)'].std()/100:.6f}" if 'O2 (%)' in cal_df.columns else "N/A",
                f"{cal_df['CO2 (%)'].std()/100:.6f}" if 'CO2 (%)' in cal_df.columns else "N/A",
                f"{cal_df['CO (%)'].std()/100:.8f}" if 'CO (%)' in cal_df.columns else "N/A",
                f"{cal_df['Stack TC (K)'].std():.1f}" if 'Stack TC (K)' in cal_df.columns else "N/A",
                f"{cal_df['DPT (Pa)'].std():.2f}" if 'DPT (Pa)' in cal_df.columns else "N/A"
            ]
            
            # Add methane rows
            if methane_slpm_cal is not None:
                cal_params.extend(['───', 'CH₄ (SLPM)', 'ṁ_CH₄ (kg/s)', 'q̇_CH₄ (kW)', 'Target (kW)'])
                cal_means.extend([
                    '',
                    f"{methane_slpm_cal:.3f}",
                    f"{m_dot_CH4_kg_s:.6e}",
                    f"{q_dot_CH4_kW:.3f}",
                    f"{target_hrr:.1f}" if target_hrr else "N/A"
                ])
                cal_stds.extend([
                    '',
                    f"{methane_slpm_std:.3f}" if methane_slpm_std else "N/A",
                    f"{m_dot_CH4_kg_s_std:.6e}" if m_dot_CH4_kg_s_std else "N/A",
                    f"{q_dot_CH4_kW_std:.3f}" if q_dot_CH4_kW_std else "N/A",
                    f"Δ {((q_dot_CH4_kW - target_hrr) / target_hrr * 100):+.1f}%" if target_hrr else "—"
                ])
            
            cal_stats = pd.DataFrame({
                'Parameter': cal_params,
                'Mean': cal_means,
                'Std Dev': cal_stds
            })
            st.dataframe(cal_stats, use_container_width=True, hide_index=True)
        
        st.divider()
        
        # --- C-Factor Calculation ---
        st.header("🔥 Calculated C-Factors")
        
        if m_dot_CH4_kg_s is None:
            st.error("Cannot calculate C-factor without methane flow rate data!")
        elif phi_iso is None:
            st.error("Cannot calculate oxygen depletion factor (φ)!")
        else:
            c_factor_iso = calculate_c_factor(
                q_dot_MW=q_dot_CH4_MW, E=E_value, X_O2_amb=X_O2_amb,
                delta_P=delta_P_cal, T_e=T_e_cal, phi=phi_iso,
                X_CO=X_CO_cal, X_O2=X_O2_cal
            )
            
            c_factor_series = cal_df['C-Factor'].dropna() if 'C-Factor' in cal_df.columns else pd.Series()
            c_factor_mean = c_factor_series.mean() if len(c_factor_series) > 0 else None
            c_factor_std = c_factor_series.std() if len(c_factor_series) > 0 else None
            c_factor_count = len(c_factor_series)
            
            co_correction_iso = 0.172 * (1 - phi_iso) * (X_CO_cal / X_O2_cal) if X_O2_cal > 0 else 0
            bracket_term_iso = (phi_iso - co_correction_iso) / (1 - phi_iso + 1.105 * phi_iso)
            flow_term_iso = sqrt(delta_P_cal / T_e_cal)
            
            ref_c_previous = get_number(metadata, 'Initial C-factor (SI units)')
            ref_c_mean = get_number(metadata, 'Mean C-factor')
            ref_c_iso = get_number(metadata, 'ISO 5660-1 C-factor')
            
            def calc_diff(calc_val, ref_val):
                if calc_val and ref_val and ref_val != 0:
                    return f"{((calc_val - ref_val) / ref_val * 100):.2f}%"
                return "N/A"
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**ISO 5660-1 Method**")
                st.caption("Average all signals, then calculate C")
                st.metric(label="C-Factor (ISO)",
                          value=f"{c_factor_iso:.6f}" if c_factor_iso else "N/A")
                
                iso_comparison = pd.DataFrame({
                    'Comparison': ['vs File ISO', 'vs File Previous'],
                    'File Value': [
                        f"{ref_c_iso:.6f}" if ref_c_iso else "N/A",
                        f"{ref_c_previous:.6f}" if ref_c_previous else "N/A"
                    ],
                    'Difference': [
                        calc_diff(c_factor_iso, ref_c_iso),
                        calc_diff(c_factor_iso, ref_c_previous)
                    ]
                })
                st.dataframe(iso_comparison, use_container_width=True, hide_index=True)
            
            with col2:
                st.markdown("**Mean Method**")
                st.caption("Calculate C at each point, then average")
                st.metric(label="C-Factor (Mean)",
                          value=f"{c_factor_mean:.6f}" if c_factor_mean else "N/A",
                          delta=f"σ = {c_factor_std:.6f}" if c_factor_std else None)
                if c_factor_count:
                    st.caption(f"Based on {c_factor_count} points")
                
                mean_comparison = pd.DataFrame({
                    'Comparison': ['vs File Mean', 'vs File Previous'],
                    'File Value': [
                        f"{ref_c_mean:.6f}" if ref_c_mean else "N/A",
                        f"{ref_c_previous:.6f}" if ref_c_previous else "N/A"
                    ],
                    'Difference': [
                        calc_diff(c_factor_mean, ref_c_mean),
                        calc_diff(c_factor_mean, ref_c_previous)
                    ]
                })
                st.dataframe(mean_comparison, use_container_width=True, hide_index=True)
            

            with st.expander("🧮 View Equations Used", expanded=False):
                st.markdown("### ASTM E1354 A1.4 Equations")
                st.markdown(r"""
                **C-Factor:**
                $$C = \frac{\dot{q}}{1.10 \cdot E \cdot X_{O_2}^{amb} \cdot \sqrt{\frac{\Delta P}{T_e}} \cdot \frac{\phi - 0.172(1-\phi)\frac{X_{CO}}{X_{O_2}}}{1 - \phi + 1.105\phi}}$$
                
                **Where:**
                - $\dot{q} = \dot{m}_{CH_4} \cdot \Delta H_c$ (HRR from methane)
                - $X_{O_2}^{amb} = (1 - X_{H_2O}^0) \cdot X_{O_2}^0$
                - $\phi = \frac{X_{O_2}^0(1 - X_{CO_2} - X_{CO}) - X_{O_2}(1 - X_{CO_2}^0)}{X_{O_2}^0(1 - X_{CO_2} - X_{CO} - X_{O_2})}$
                """)
            
            st.divider()
            
            # --- Export Results ---
            st.header("💾 Export Results")
            
            adj_summary = ""
            if any_adjustments:
                adj_summary = "\nADJUSTMENTS APPLIED:\n"
                for param, vals in adjustments['parameters'].items():
                    adj_summary += f"  {param}: {vals['file']} → {vals['current']}\n"
                for col, adj in adjustments['columns'].items():
                    adj_summary += f"  {col}: Scale={adj['scale']:.4f}, Shift={adj['shift']:.4f}\n"
            
            summary = f"""C-Factor Calibration Summary (ASTM E1354 A1.4)
{'='*60}

Test: {get_string(metadata, 'Filename')}
Date: {get_string(metadata, 'Date of test')}
{adj_summary}
RESULTS:
  C-Factor (ISO):  {f"{c_factor_iso:.6f}" if c_factor_iso else 'N/A'}
  C-Factor (Mean): {f"{c_factor_mean:.6f}" if c_factor_mean else 'N/A'} (σ = {f"{c_factor_std:.6f}" if c_factor_std else 'N/A'})

FILE VALUES:
  Previous: {ref_c_previous}
  Mean:     {ref_c_mean}
  ISO:      {ref_c_iso}

PARAMETERS:
  Ambient: {amb_temp_C}°C, {amb_pressure_Pa} Pa, {rel_humid_pct}% RH
  E: {E_value} MJ/kg
  Delays: O2={o2_delay}s, CO2={co2_delay}s, CO={co_delay}s

REGIONS:
  Baseline:    {baseline_start}-{baseline_end}s ({n_baseline} pts)
  Calibration: {cal_start}-{cal_end}s ({n_calibration} pts)

CALIBRATION AVERAGES:
  CH4: {f"{methane_slpm_cal:.4f}" if methane_slpm_cal else "N/A"} SLPM → {f"{q_dot_CH4_kW:.3f}" if q_dot_CH4_kW else "N/A"} kW
  X_O2: {X_O2_cal:.6f}, X_CO2: {X_CO2_cal:.6f}, X_CO: {X_CO_cal:.8f}
  T_e: {T_e_cal:.1f} K, ΔP: {delta_P_cal:.2f} Pa
  φ: {f"{phi_iso:.6f}" if phi_iso else "N/A"}, X_O2^amb: {X_O2_amb:.6f}
"""
            
            col1, col2 = st.columns(2)
            with col1:
                st.download_button(label="📥 Download Summary (TXT)", data=summary,
                                   file_name="c_factor_summary.txt", mime="text/plain")
            with col2:
                import json
                export_json = {
                    "file": get_string(metadata, 'Filename'),
                    "adjustments": adjustments if any_adjustments else None,
                    "results": {"iso": c_factor_iso, "mean": c_factor_mean, "std": c_factor_std},
                    "file_values": {"previous": ref_c_previous, "mean": ref_c_mean, "iso": ref_c_iso}
                }
                st.download_button(label="📥 Download Summary (JSON)",
                                   data=json.dumps(export_json, indent=2, default=str),
                                   file_name=f"c_factor_summary_{shortname.replace('.ftt', '')}.json", mime="application/json")

else:
    st.info("👆 Please upload a calibration file or enter a file path to begin.")
    
    # Show empty sidebar message when no data loaded
    with st.sidebar:
        st.info("Load a calibration file to access adjustment controls.")

st.divider()
st.markdown("#### Notes")
readme = SCRIPT_DIR / "README.md"
section_title = "### CFactor Check"

with open(readme, "r", encoding="utf-8") as f:
    lines = f.readlines()

start_idx, end_idx = None, None
for i, line in enumerate(lines):
    if line.strip() == section_title:
        start_idx = i + 1
        break

if start_idx is not None:
    for j in range(start_idx + 1, len(lines)):
        if lines[j].startswith("### ") or lines[j].startswith("## "):
            end_idx = j
            break
    if end_idx is None:
        end_idx = len(lines)
    subsection = "".join(lines[start_idx:end_idx])
    st.markdown(subsection)