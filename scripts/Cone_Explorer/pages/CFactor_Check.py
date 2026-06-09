# -*- coding: utf-8 -*-
"""C-Factor Calibration Calculator based on ASTM E1354."""

import sys

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from math import sqrt
from pathlib import Path
import math
from datetime import datetime, timedelta
import os

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))
from Cone_Explorer.const import (
    SCRIPT_DIR, CALIB_DATA
)


st.set_page_config(page_title="C-Factor Calculator", page_icon="🔥", layout="wide")

st.title("🔥 C-Factor Calibration Calculator")
st.markdown("Based on **ASTM E1354 Appendix A1.4**")


# --- Constants ---
DELTA_HC_METHANE = 50.01  # MJ/kg - Heat of combustion of methane
E_DEFAULT = 12.54  # MJ/kg - ΔHc/r_0 for methane

# Clausius-Clapeyron constants for water
P_REF = 101325  # Pa - Reference pressure (1 atm)
T_REF = 373.15  # K - Reference temperature (boiling point)
DELTA_H_VAP = 40.7e3  # J/mol - Heat of vaporization of water
R_GAS = 8.314  # J/(mol·K) - Universal gas constant


# --- Helper Functions ---

def find_calib_folder(base_path, instrument_name):
    """Find the Calib subfolder for an instrument, handling case sensitivity."""
    instrument_folder = base_path / instrument_name
    
    for calib_name in ["Calib", "calib", "CALIB"]:
        calib_path = instrument_folder / calib_name
        try:
            if calib_path.exists():
                return calib_path
        except:
            pass
        try:
            if os.path.exists(str(calib_path)):
                return calib_path
        except:
            pass
    
    try:
        if instrument_folder.exists():
            for item in instrument_folder.iterdir():
                if item.is_dir() and item.name.lower() == "calib":
                    return item
    except:
        pass
    
    try:
        for item_name in os.listdir(str(instrument_folder)):
            item_path = instrument_folder / item_name
            if os.path.isdir(str(item_path)) and item_name.lower() == "calib":
                return item_path
    except:
        pass
    
    return None


def find_file_in_folder(folder_path, target_name, case_insensitive=True):
    """Find a file in a folder."""
    target_lower = target_name.lower() if case_insensitive else target_name
    
    direct_path = folder_path / target_name
    try:
        if direct_path.exists():
            return direct_path
    except:
        pass
    
    try:
        if os.path.exists(str(direct_path)):
            return direct_path
    except:
        pass
    
    try:
        for item in folder_path.iterdir():
            item_name = item.name
            compare_name = item_name.lower() if case_insensitive else item_name
            if compare_name == target_lower:
                return item
    except:
        pass
    
    try:
        for item_name in os.listdir(str(folder_path)):
            compare_name = item_name.lower() if case_insensitive else item_name
            if compare_name == target_lower:
                return folder_path / item_name
    except:
        pass
    
    return None


def calculate_X_H2O(amb_temp_C, rel_humid_pct, amb_pressure_Pa):
    """Calculate water vapor mole fraction from ambient conditions."""
    T_K = amb_temp_C + 273.15
    cc_const = DELTA_H_VAP / R_GAS
    cc_offset = cc_const / T_REF
    p_sat_water = P_REF * math.exp(cc_offset - cc_const / T_K)
    
    p_h2o = (rel_humid_pct / 100) * p_sat_water
    X_H2O = p_h2o / amb_pressure_Pa
    return X_H2O


def get_psat_water(temp_C):
    """Calculate saturation pressure of water using Clausius-Clapeyron equation."""
    T_K = temp_C + 273.15
    cc_const = DELTA_H_VAP / R_GAS
    cc_offset = cc_const / T_REF
    return P_REF * math.exp(cc_offset - cc_const / T_K)


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


def parse_c_logs(file_path):
    """
    Parse C-Logs.CSV file which has a simple CSV format:
    date,time,c-factor,filepath
    
    Returns a DataFrame with columns: Date, Time, C-Factor, Filepath, DateTime
    """
    records = []
    
    content = None
    for encoding in ['cp1252', 'utf-8', 'latin-1', 'iso-8859-1']:
        try:
            with open(file_path, 'r', encoding=encoding) as f:
                content = f.read()
            break
        except Exception:
            continue
    
    if content is None:
        with open(file_path, 'rb') as f:
            content = f.read().decode('cp1252', errors='ignore')
    
    lines = content.strip().split('\n')
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        parts = line.split(',')
        
        if len(parts) >= 4:
            try:
                date_str = parts[0].strip()
                time_str = parts[1].strip()
                c_factor = float(parts[2].strip())
                filepath = parts[3].strip()
                
                filename = filepath.replace('\\', '/').split('/')[-1]
                
                records.append({
                    'Date': date_str,
                    'Time': time_str,
                    'C-Factor': c_factor,
                    'Filepath': filename
                })
            except (ValueError, IndexError):
                continue
    
    df = pd.DataFrame(records)
    
    if len(df) == 0:
        return df
    
    def parse_date(date_str):
        for fmt in ['%d/%m/%Y', '%m/%d/%Y', '%d/%m/%y', '%m/%d/%y', '%Y-%m-%d']:
            try:
                return pd.to_datetime(date_str, format=fmt)
            except:
                continue
        try:
            return pd.to_datetime(date_str, dayfirst=True)
        except:
            return pd.NaT
    
    df['DateTime'] = df['Date'].apply(parse_date)
    df = df.dropna(subset=['C-Factor', 'DateTime'])
    df = df.sort_values('DateTime')
    df = df.drop_duplicates(subset=['Filepath'], keep='last')
    
    return df


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


def extract_calibration_parameters(file_path, calib_folder, default_cal_start=305, default_cal_end=420):
    """
    Extract calibration region average parameters from a single test file.
    Returns a dictionary of parameters or None if file cannot be processed.
    """
    try:
        full_path = find_file_in_folder(calib_folder, file_path, case_insensitive=True)
        if full_path is None:
            return None
        
        metadata, data = parse_ftt_file(file_path=full_path, encoding='cp1252')
        
        # Rename time column if needed
        time_col = data.columns[0]
        if time_col != 'Time (s)':
            data = data.rename(columns={time_col: 'Time (s)'})
        
        # Find methane column
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
        
        # Get burner timing from metadata
        burner_on_s = int(get_number(metadata, 'Burner on (s)', 65) or 65)
        burner_off_s = int(get_number(metadata, 'Burner off (s)', 425) or 425)
        
        # Calculate calibration region
        cal_start = burner_on_s + 240
        cal_end = burner_off_s - 5
        
        # Get baseline region
        baseline_start = 10
        baseline_end = burner_on_s - 5
        
        # Filter data for regions
        baseline_mask = (data['Time (s)'] >= baseline_start) & (data['Time (s)'] <= baseline_end)
        calibration_mask = (data['Time (s)'] >= cal_start) & (data['Time (s)'] <= cal_end)
        
        baseline_df = data[baseline_mask]
        cal_df = data[calibration_mask]
        
        if len(cal_df) == 0 or len(baseline_df) == 0:
            return None
        
        # Extract ambient conditions
        amb_temp_C = get_number(metadata, 'Ambient temperature (°C)', 20.0)
        if amb_temp_C is None:
            amb_temp_C = get_number(metadata, 'Ambient temperature', 20.0) or 20.0
        amb_pressure_Pa = get_number(metadata, 'Barometric pressure (Pa)', 101325.0) or 101325.0
        rel_humid_pct = get_number(metadata, 'Relative humidity (%)', 50.0) or 50.0
        
        # Calculate baseline (initial) values
        X_O2_0 = baseline_df['O2 (%)'].mean() / 100 if 'O2 (%)' in baseline_df.columns else 0.2095
        X_CO2_0 = baseline_df['CO2 (%)'].mean() / 100 if 'CO2 (%)' in baseline_df.columns else 0.0004
        X_CO_0 = baseline_df['CO (%)'].mean() / 100 if 'CO (%)' in baseline_df.columns else 0.0
        
        # Calculate calibration averages
        params = {
            'File': file_path,
            'Date': get_string(metadata, 'Date of test'),
            'Amb Temp (°C)': amb_temp_C,
            'Pressure (Pa)': amb_pressure_Pa,
            'RH (%)': rel_humid_pct,
            # Initial (baseline) volume fractions
            'X_O2_0 (%)': X_O2_0 * 100,
            'X_CO2_0 (%)': X_CO2_0 * 100,
            'X_CO_0 (%)': X_CO_0 * 100,
            # Calibration region values
            'O2 (%)': cal_df['O2 (%)'].mean() if 'O2 (%)' in cal_df.columns else None,
            'CO2 (%)': cal_df['CO2 (%)'].mean() if 'CO2 (%)' in cal_df.columns else None,
            'CO (%)': cal_df['CO (%)'].mean() if 'CO (%)' in cal_df.columns else None,
            'Stack TC (K)': cal_df['Stack TC (K)'].mean() if 'Stack TC (K)' in cal_df.columns else None,
            'DPT (Pa)': cal_df['DPT (Pa)'].mean() if 'DPT (Pa)' in cal_df.columns else None,
            'Target HRR (kW)': get_number(metadata, 'HRR level (kW)'),
            'Burner On (s)': burner_on_s,
            'Burner Off (s)': burner_off_s,
            'Cal Start (s)': cal_start,
            'Cal End (s)': cal_end,
            'N Points': len(cal_df),
        }
        
        # Add methane data if available
        if methane_col and methane_col in cal_df.columns:
            methane_slpm = cal_df[methane_col].mean()
            m_dot_kg_s = slpm_to_kg_s_methane(methane_slpm)
            hrr_kw = calculate_hrr_methane(m_dot_kg_s) * 1000
            params['CH4 (SLPM)'] = methane_slpm
            params['ṁ_CH4 (kg/s)'] = m_dot_kg_s
            params['HRR (kW)'] = hrr_kw
        
        # Calculate phi
        if params['O2 (%)'] and params['CO2 (%)'] and params['CO (%)']:
            X_O2 = params['O2 (%)'] / 100
            X_CO2 = params['CO2 (%)'] / 100
            X_CO = params['CO (%)'] / 100
            phi = calculate_odf(X_O2, X_CO2, X_CO, X_O2_0, X_CO2_0)
            params['φ (ODF)'] = phi
        
        # Get file C-factor values
        params['File C-Factor (Mean)'] = get_number(metadata, 'Mean C-factor')
        params['File C-Factor (ISO)'] = get_number(metadata, 'ISO 5660-1 C-factor')
        
        return params
        
    except Exception as e:
        return None

# --- Main App ---

# Initialize session state
if 'selected_instrument' not in st.session_state:
    st.session_state.selected_instrument = None
if 'selected_file' not in st.session_state:
    st.session_state.selected_file = None
if 'c_logs_df' not in st.session_state:
    st.session_state.c_logs_df = None
if 'excluded_points' not in st.session_state:
    st.session_state.excluded_points = set()
if 'all_test_params' not in st.session_state:
    st.session_state.all_test_params = None


# --- Step 1: Instrument Selection ---
st.header("📁 Select Instrument")

# Get available instrument folders
instrument_folders = []
if CALIB_DATA.exists() or os.path.exists(str(CALIB_DATA)):
    try:
        for folder in CALIB_DATA.iterdir():
            if folder.is_dir():
                calib_subfolder = find_calib_folder(CALIB_DATA, folder.name)
                if calib_subfolder is not None:
                    instrument_folders.append(folder.name)
    except Exception:
        try:
            for folder_name in os.listdir(str(CALIB_DATA)):
                folder_path = CALIB_DATA / folder_name
                if os.path.isdir(str(folder_path)):
                    calib_subfolder = find_calib_folder(CALIB_DATA, folder_name)
                    if calib_subfolder is not None:
                        instrument_folders.append(folder_name)
        except Exception:
            pass

if not instrument_folders:
    st.error(f"No instrument folders found in {CALIB_DATA}")
    st.info("Expected folder structure: CALIB_DATA/[Instrument]/Calib/")
    st.stop()

col1, col2 = st.columns([2, 1])
with col1:
    selected_instrument = st.selectbox(
        "Select Instrument",
        options=instrument_folders,
        index=instrument_folders.index(st.session_state.selected_instrument) if st.session_state.selected_instrument in instrument_folders else 0,
        help="Select the cone calorimeter instrument"
    )

if selected_instrument != st.session_state.selected_instrument:
    st.session_state.selected_instrument = selected_instrument
    st.session_state.selected_file = None
    st.session_state.c_logs_df = None
    st.session_state.excluded_points = set()
    st.session_state.all_test_params = None
    reset_adjustments()

# --- Step 2: Load C-Logs and Display History ---
if selected_instrument:
    calib_folder = find_calib_folder(CALIB_DATA, selected_instrument)
    
    if calib_folder is None:
        st.error(f"Calib folder not found for {selected_instrument}")
        st.stop()
    
    c_logs_path = find_file_in_folder(calib_folder, "C-Logs.CSV", case_insensitive=True)
    
    if c_logs_path is None:
        st.error(f"C-Logs.CSV not found in {calib_folder}")
        st.stop()
    
    # Load C-Logs
    if st.session_state.c_logs_df is None:
        try:
            st.session_state.c_logs_df = parse_c_logs(c_logs_path)
        except Exception as e:
            st.error(f"Error loading C-Logs.CSV: {e}")
            st.stop()
    
    c_logs_df = st.session_state.c_logs_df.copy()
    
    if len(c_logs_df) == 0:
        st.warning("No valid calibration records found in C-Logs.CSV")
        st.stop()
    
    st.success(f"✅ Loaded {len(c_logs_df)} calibration records from {selected_instrument}")
    
    # --- Date Range Selection ---
    st.header("📅 Date Range & Point Selection")
    
    min_date = c_logs_df['DateTime'].min().date()
    max_date = c_logs_df['DateTime'].max().date()
    
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col1:
        date_range_start = st.date_input(
            "Start Date",
            value=min_date,
            min_value=min_date,
            max_value=max_date,
            key="date_range_start"
        )
    
    with col2:
        date_range_end = st.date_input(
            "End Date",
            value=max_date,
            min_value=min_date,
            max_value=max_date,
            key="date_range_end"
        )
    
    with col3:
        preset = st.selectbox(
            "Quick Presets",
            options=["Custom", "Last 30 days", "Last 90 days", "Last 6 months", "Last year", "All time"],
            index=0,
            key="date_preset"
        )
        
        if preset != "Custom":
            if preset == "Last 30 days":
                date_range_start = max_date - timedelta(days=30)
            elif preset == "Last 90 days":
                date_range_start = max_date - timedelta(days=90)
            elif preset == "Last 6 months":
                date_range_start = max_date - timedelta(days=180)
            elif preset == "Last year":
                date_range_start = max_date - timedelta(days=365)
            elif preset == "All time":
                date_range_start = min_date
            date_range_end = max_date
    
    # Filter by date range
    date_mask = (c_logs_df['DateTime'].dt.date >= date_range_start) & (c_logs_df['DateTime'].dt.date <= date_range_end)
    c_logs_filtered = c_logs_df[date_mask].copy()
    
    c_logs_filtered = c_logs_filtered.reset_index(drop=True)
    c_logs_filtered['PointID'] = c_logs_filtered['Filepath']
    
    c_logs_filtered['Included'] = ~c_logs_filtered['PointID'].isin(st.session_state.excluded_points)
    
    included_df = c_logs_filtered[c_logs_filtered['Included']]
    
    if len(included_df) > 0:
        c_mean = included_df['C-Factor'].mean()
        c_std = included_df['C-Factor'].std()
        c_min = included_df['C-Factor'].min()
        c_max = included_df['C-Factor'].max()
    else:
        c_mean = c_std = c_min = c_max = 0
    
    # --- Plot C-Factor History ---
    st.header("📈 C-Factor History")
    
    fig = go.Figure()
    
    included_plot = c_logs_filtered[c_logs_filtered['Included']]
    if len(included_plot) > 0:
        fig.add_trace(go.Scatter(
            x=included_plot['DateTime'],
            y=included_plot['C-Factor'],
            mode='lines+markers',
            name='Included',
            line=dict(color='blue', width=1),
            marker=dict(size=10, color='blue', symbol='circle'),
            hovertemplate=(
                '<b>Date:</b> %{x|%Y-%m-%d}<br>'
                '<b>C-Factor:</b> %{y:.6f}<br>'
                '<b>File:</b> %{customdata}<br>'
                '<b>Status:</b> Included<extra></extra>'
            ),
            customdata=included_plot['Filepath']
        ))
    
    excluded_plot = c_logs_filtered[~c_logs_filtered['Included']]
    if len(excluded_plot) > 0:
        fig.add_trace(go.Scatter(
            x=excluded_plot['DateTime'],
            y=excluded_plot['C-Factor'],
            mode='markers',
            name='Excluded',
            marker=dict(size=10, color='red', symbol='x'),
            hovertemplate=(
                '<b>Date:</b> %{x|%Y-%m-%d}<br>'
                '<b>C-Factor:</b> %{y:.6f}<br>'
                '<b>File:</b> %{customdata}<br>'
                '<b>Status:</b> EXCLUDED<extra></extra>'
            ),
            customdata=excluded_plot['Filepath']
        ))

    fig.update_layout(
        title=f"C-Factor History - {selected_instrument} ({date_range_start} to {date_range_end})",
        xaxis_title="Date",
        yaxis_title="C-Factor",
        height=450,
        hovermode='closest',
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("Mean C-Factor", f"{c_mean:.6f}")
    with col2:
        st.metric("Std Dev", f"{c_std:.6f}")
    with col3:
        st.metric("Min", f"{c_min:.6f}")
    with col4:
        st.metric("Max", f"{c_max:.6f}")
    with col5:
        st.metric("Points", f"{len(included_df)}/{len(c_logs_filtered)}")
    
    # --- Point Selection Interface ---
    st.subheader("🎯 Overview of Calibrations")
    
    col1, col2 = st.columns([3, 1])
    
    with col2:
        st.markdown("**Quick Actions**")
        if st.button("✅ Include All", use_container_width=True):
            st.session_state.excluded_points = set()
            st.rerun()
        
        if st.button("❌ Exclude All", use_container_width=True):
            st.session_state.excluded_points = set(c_logs_filtered['PointID'].tolist())
            st.rerun()
        
        if st.button("🔄 Exclude Outliers (>2σ)", use_container_width=True):
            if len(included_df) > 0:
                outlier_mask = (c_logs_filtered['C-Factor'] > c_mean + 2*c_std) | (c_logs_filtered['C-Factor'] < c_mean - 2*c_std)
                outlier_ids = c_logs_filtered[outlier_mask]['PointID'].tolist()
                st.session_state.excluded_points.update(outlier_ids)
                st.rerun()
        
        if st.button("🔄 Exclude Outliers (>3σ)", use_container_width=True):
            if len(included_df) > 0:
                outlier_mask = (c_logs_filtered['C-Factor'] > c_mean + 3*c_std) | (c_logs_filtered['C-Factor'] < c_mean - 3*c_std)
                outlier_ids = c_logs_filtered[outlier_mask]['PointID'].tolist()
                st.session_state.excluded_points.update(outlier_ids)
                st.rerun()
    
    with col1:
        display_df = c_logs_filtered[['DateTime', 'C-Factor', 'Filepath', 'Included']].copy()
        display_df['DateTime'] = display_df['DateTime'].dt.strftime('%Y-%m-%d %H:%M')
        display_df = display_df.rename(columns={
            'DateTime': 'Date/Time',
            'C-Factor': 'C-Factor',
            'Filepath': 'File',
            'Included': '✓ Include'
        })
        
        edited_df = st.data_editor(
            display_df,
            column_config={
                "Date/Time": st.column_config.TextColumn("Date/Time", disabled=True),
                "C-Factor": st.column_config.NumberColumn("C-Factor", format="%.6f", disabled=True),
                "File": st.column_config.TextColumn("File", disabled=True),
                "✓ Include": st.column_config.CheckboxColumn("✓ Include", default=True)
            },
            hide_index=True,
            use_container_width=True,
            height=300
        )
        
        new_excluded = set()
        for idx, row in edited_df.iterrows():
            if not row['✓ Include']:
                point_id = c_logs_filtered.iloc[idx]['PointID']
                new_excluded.add(point_id)
        
        if new_excluded != st.session_state.excluded_points:
            st.session_state.excluded_points = new_excluded
            st.rerun()
    
    st.divider()
    
   # --- All Tests Parameter Summary ---
    st.header("📊 Parameters Used for Calibration")
    
    with st.expander("View/Load Calibration Parameters for All Tests in Date Range", expanded=False):
        
        col1, col2 = st.columns([1, 3])
        
        with col1:
            if st.button("🔄 Load/Refresh All Test Data", use_container_width=True):
                with st.spinner("Loading calibration data from all test files..."):
                    all_params = []
                    progress_bar = st.progress(0)
                    
                    files_to_process = c_logs_filtered['Filepath'].tolist()
                    total_files = len(files_to_process)
                    
                    for i, filepath in enumerate(files_to_process):
                        params = extract_calibration_parameters(filepath, calib_folder)
                        if params:
                            # Add C-Factor from C-Logs
                            c_logs_row = c_logs_filtered[c_logs_filtered['Filepath'] == filepath]
                            if len(c_logs_row) > 0:
                                params['C-Logs C-Factor'] = c_logs_row.iloc[0]['C-Factor']
                                params['DateTime'] = c_logs_row.iloc[0]['DateTime']
                                params['Included'] = c_logs_row.iloc[0]['Included']
                            all_params.append(params)
                        
                        progress_bar.progress((i + 1) / total_files)
                    
                    if all_params:
                        st.session_state.all_test_params = pd.DataFrame(all_params)
                    else:
                        st.session_state.all_test_params = pd.DataFrame()
                    
                    progress_bar.empty()
                
                st.success(f"Loaded data from {len(all_params)} of {total_files} files")
                st.rerun()
        
        with col2:
            if st.session_state.all_test_params is not None and len(st.session_state.all_test_params) > 0:
                st.caption(f"Data loaded for {len(st.session_state.all_test_params)} tests")
            else:
                st.caption("Click 'Load/Refresh' to extract parameters from all test files")
        
        if st.session_state.all_test_params is not None and len(st.session_state.all_test_params) > 0:
            params_df = st.session_state.all_test_params.copy()
            
            # Column selection
            available_cols = params_df.columns.tolist()
            default_cols = ['File', 'DateTime', 'C-Logs C-Factor', 'O2 (%)', 'CO2 (%)', 'CO (%)', 
                           'Stack TC (K)', 'DPT (Pa)', 'CH4 (SLPM)', 'ṁ_CH4 (kg/s)', 'HRR (kW)', 'Included']
            default_cols = [c for c in default_cols if c in available_cols]
            
            selected_cols = st.multiselect(
                "Select columns to display",
                options=available_cols,
                default=default_cols,
                key="param_cols_select"
            )
            
            if selected_cols:
                display_params_df = params_df[selected_cols].copy()
                
                # Format datetime if present
                if 'DateTime' in display_params_df.columns:
                    display_params_df['DateTime'] = pd.to_datetime(display_params_df['DateTime']).dt.strftime('%Y-%m-%d %H:%M')
                
                # Format mass flow rate in scientific notation (3 sig figs)
                if 'ṁ_CH4 (kg/s)' in display_params_df.columns:
                    display_params_df['ṁ_CH4 (kg/s)'] = display_params_df['ṁ_CH4 (kg/s)'].apply(
                        lambda x: f"{x:.3e}" if pd.notna(x) else "N/A"
                    )
                
                # Sort by datetime
                if 'DateTime' in display_params_df.columns:
                    display_params_df = display_params_df.sort_values('DateTime', ascending=False)
                
                st.dataframe(
                    display_params_df,
                    use_container_width=True,
                    hide_index=True,
                    height=400
                )
                
                # Statistics for included tests only
                st.subheader("📈 Statistics (Included Tests Only)")
                
                included_params = params_df[params_df['Included'] == True] if 'Included' in params_df.columns else params_df
                
                numeric_cols = ['C-Logs C-Factor', 'X_O2_0 (%)', 'X_CO2_0 (%)', 'X_CO_0 (%)',
                               'O2 (%)', 'CO2 (%)', 'CO (%)', 'Stack TC (K)', 'DPT (Pa)', 
                               'CH4 (SLPM)', 'ṁ_CH4 (kg/s)', 'HRR (kW)', 'φ (ODF)', 
                               'Amb Temp (°C)', 'Pressure (Pa)', 'RH (%)']
                numeric_cols = [c for c in numeric_cols if c in included_params.columns]
                
                if numeric_cols:
                    stats_data = []
                    for col in numeric_cols:
                        col_data = included_params[col].dropna()
                        if len(col_data) > 0:
                            stats_data.append({
                                'Parameter': col,
                                'Mean': col_data.mean(),
                                'Std Dev': col_data.std(),
                                'Min': col_data.min(),
                                'Max': col_data.max(),
                                'N': len(col_data)
                            })
                    
                    if stats_data:
                        stats_df = pd.DataFrame(stats_data)
                        
                        # Unified formatting function
                        def format_value(val, param):
                            if pd.isna(val):
                                return "N/A"
                            # C-Factor: 6 decimal places
                            if 'C-Factor' in param:
                                return f"{val:.6f}"
                            # Mass flow rate: scientific notation, 3 sig figs
                            elif 'ṁ_CH4' in param:
                                return f"{val:.3e}"
                            # Gas concentrations: 4 decimal places
                            elif param in ['O2 (%)', 'CO2 (%)', 'CO (%)', 'X_O2_0 (%)', 'X_CO2_0 (%)', 'X_CO_0 (%)']:
                                return f"{val:.4f}"
                            # ODF: 4 decimal places
                            elif 'φ' in param or 'ODF' in param:
                                return f"{val:.4f}"
                            # Temperature, pressure: 1 decimal place
                            elif param in ['Stack TC (K)', 'DPT (Pa)', 'Pressure (Pa)', 'Amb Temp (°C)', 'RH (%)']:
                                return f"{val:.1f}"
                            # HRR and flow rates: 3 decimal places
                            elif param in ['HRR (kW)', 'CH4 (SLPM)']:
                                return f"{val:.3f}"
                            # Default: 3 decimal places
                            else:
                                return f"{val:.3f}"
                        
                        for col in ['Mean', 'Std Dev', 'Min', 'Max']:
                            stats_df[col] = stats_df.apply(
                                lambda row: format_value(row[col], row['Parameter']), axis=1
                            )
                        
                        st.dataframe(stats_df, use_container_width=True, hide_index=True)
                
                # Download buttons
                col1, col2 = st.columns(2)
                with col1:
                    csv_data = params_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download All Parameters (CSV)",
                        data=csv_data,
                        file_name=f"calibration_parameters_{selected_instrument}.csv",
                        mime="text/csv"
                    )
                with col2:
                    if len(included_params) > 0:
                        csv_data_included = included_params.to_csv(index=False)
                        st.download_button(
                            label="📥 Download Included Only (CSV)",
                            data=csv_data_included,
                            file_name=f"calibration_parameters_{selected_instrument}_included.csv",
                            mime="text/csv"
                        )
    st.divider()
    
    # --- Step 3: Select Calibration File ---
    st.header("📂 Single Calibration Detailed Analysis")
    
    c_logs_filtered['SelectLabel'] = c_logs_filtered.apply(
        lambda row: f"{'❌ ' if not row['Included'] else ''}{row['DateTime'].strftime('%Y-%m-%d')} | C={row['C-Factor']:.6f} | {row['Filepath']}",
        axis=1
    )
    
    selection_df = c_logs_filtered.sort_values('DateTime', ascending=False)
    
    selected_label = st.selectbox(
        "Select a calibration test to analyze",
        options=selection_df['SelectLabel'].tolist(),
        index=0,
        help="Select a calibration from the history to view details"
    )
    
    selected_row = selection_df[selection_df['SelectLabel'] == selected_label].iloc[0]
    selected_filepath = selected_row['Filepath']
    
    file_path = find_file_in_folder(calib_folder, selected_filepath, case_insensitive=True)
    
    if file_path is None:
        st.error(f"Calibration file not found: {selected_filepath}")
        st.info(f"Looking in: {calib_folder}")
        st.stop()
    
    if str(file_path) != st.session_state.selected_file:
        st.session_state.selected_file = str(file_path)
        reset_adjustments()

    st.divider()
    
    # --- Load and Process Selected File ---
    try:
        metadata, data = parse_ftt_file(file_path=file_path, encoding='cp1252')
        shortname = file_path.name
        st.success(f"✅ Loaded: {shortname} ({len(data)} data points)")
    except Exception as e:
        st.error(f"Error reading file: {e}")
        st.stop()

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
    
    burner_on_s = int(get_number(metadata, 'Burner on (s)', 65) or 65)
    burner_off_s = int(get_number(metadata, 'Burner off (s)', 425) or 425)
    
    default_amb_temp = get_number(metadata, 'Ambient temperature (°C)', 20.0)
    if default_amb_temp is None:
        default_amb_temp = get_number(metadata, 'Ambient temperature', 20.0)
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
        
        adjustments = {
            'parameters': {},
            'columns': {}
        }
        
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
            value=min(default_cal_start, time_max), step=1,
            key="region_cal_start"
        )
        cal_end = st.number_input(
            "End (s)", min_value=0, max_value=time_max,
            value=min(default_cal_end, time_max), step=1,
            key="region_cal_end"
        )
        
        st.divider()
        
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
            adjustments['parameters']['O₂ Delay (s)'] = {'file': default_o2_delay, 'current': o2_delay}
        if co2_delay != default_co2_delay:
            adjustments['parameters']['CO₂ Delay (s)'] = {'file': default_co2_delay, 'current': co2_delay}
        if co_delay != default_co_delay:
            adjustments['parameters']['CO Delay (s)'] = {'file': default_co_delay, 'current': co_delay}
        
        st.divider()
        
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
    
    # --- Calculate derived values ---
    X_H2O_initial = calculate_X_H2O(amb_temp_C, rel_humid_pct, amb_pressure_Pa)
    
    # --- Apply Adjustments to Data ---
    data_adjusted = data.copy()
    
    for col_key in col_keys:
        if col_key in adjustments['columns']:
            adj = adjustments['columns'][col_key]
            data_adjusted[col_key] = (data_adjusted[col_key] * adj['scale']) + adj['shift']
    
    if methane_col:
        data_adjusted['m_dot_CH4 (kg/s)'] = data_adjusted[methane_col].apply(
            lambda x: slpm_to_kg_s_methane(x) if pd.notna(x) else np.nan
        )
        data_adjusted['HRR_CH4 (MW)'] = data_adjusted['m_dot_CH4 (kg/s)'].apply(
            lambda x: calculate_hrr_methane(x) if pd.notna(x) else np.nan
        )
        data_adjusted['HRR_CH4 (kW)'] = data_adjusted['HRR_CH4 (MW)'] * 1000
    
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
    
    baseline_mask = (data_corrected['Time (s)'] >= baseline_start) & (data_corrected['Time (s)'] <= baseline_end)
    calibration_mask = (data_corrected['Time (s)'] >= cal_start) & (data_corrected['Time (s)'] <= cal_end)
    
    n_baseline = baseline_mask.sum()
    n_calibration = calibration_mask.sum()
    
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
    
    if methane_col:
        data_corrected['C-Factor'] = data_corrected.apply(
            lambda row: calculate_c_factor_for_row(row, X_O2_0, X_CO2_0, X_O2_amb, E_value, methane_col),
            axis=1
        )
        cal_df = data_corrected[calibration_mask]
        baseline_df = data_corrected[baseline_mask]
    
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
                    c_min_plot = c_factor_cal_data.min()
                    c_max_plot = c_factor_cal_data.max()
                    c_range = c_max_plot - c_min_plot if c_max_plot != c_min_plot else 0.001
                    fig.update_yaxes(range=[c_min_plot - 0.1 * c_range, c_max_plot + 0.1 * c_range], row=row, col=col)
        
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
            c_factor_std_val = c_factor_series.std() if len(c_factor_series) > 0 else None
            c_factor_count = len(c_factor_series)
            
            ref_c_previous = get_number(metadata, 'Initial C-factor (SI units)')
            ref_c_mean = get_number(metadata, 'Mean C-factor')
            ref_c_iso = get_number(metadata, 'ISO 5660-1 C-factor')
            
            c_logs_c_factor = selected_row['C-Factor']
            
            historical_c_mean = c_mean if len(included_df) > 0 else None
            historical_c_std = c_std if len(included_df) > 0 else None
            
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
                    'Comparison': ['vs C-Logs', 'vs File ISO', 'vs File Previous', 'vs Historical Mean'],
                    'Reference Value': [
                        f"{c_logs_c_factor:.6f}" if c_logs_c_factor else "N/A",
                        f"{ref_c_iso:.6f}" if ref_c_iso else "N/A",
                        f"{ref_c_previous:.6f}" if ref_c_previous else "N/A",
                        f"{historical_c_mean:.6f} (±{historical_c_std:.6f})" if historical_c_mean else "N/A"
                    ],
                    'Difference': [
                        calc_diff(c_factor_iso, c_logs_c_factor),
                        calc_diff(c_factor_iso, ref_c_iso),
                        calc_diff(c_factor_iso, ref_c_previous),
                        calc_diff(c_factor_iso, historical_c_mean)
                    ]
                })
                st.dataframe(iso_comparison, use_container_width=True, hide_index=True)
            
            with col2:
                st.markdown("**Mean Method**")
                st.caption("Calculate C at each point, then average")
                st.metric(label="C-Factor (Mean)",
                          value=f"{c_factor_mean:.6f}" if c_factor_mean else "N/A",
                          delta=f"σ = {c_factor_std_val:.6f}" if c_factor_std_val else None)
                if c_factor_count:
                    st.caption(f"Based on {c_factor_count} points")
                
                mean_comparison = pd.DataFrame({
                    'Comparison': ['vs C-Logs', 'vs File Mean', 'vs File Previous', 'vs Historical Mean'],
                    'Reference Value': [
                        f"{c_logs_c_factor:.6f}" if c_logs_c_factor else "N/A",
                        f"{ref_c_mean:.6f}" if ref_c_mean else "N/A",
                        f"{ref_c_previous:.6f}" if ref_c_previous else "N/A",
                        f"{historical_c_mean:.6f} (±{historical_c_std:.6f})" if historical_c_mean else "N/A"
                    ],
                    'Difference': [
                        calc_diff(c_factor_mean, c_logs_c_factor),
                        calc_diff(c_factor_mean, ref_c_mean),
                        calc_diff(c_factor_mean, ref_c_previous),
                        calc_diff(c_factor_mean, historical_c_mean)
                    ]
                })
                st.dataframe(mean_comparison, use_container_width=True, hide_index=True)
            
            st.divider()
            
            # --- View Equations & Calculation Breakdown ---
            with st.expander("📐 View Equations & Calculation Breakdown"):
                st.markdown("### Step-by-Step C-Factor Calculation (ISO Method)")
                
                st.markdown("---")
                st.markdown("#### 1. Water Vapor Mole Fraction")
                st.markdown("""
                Using Clausius-Clapeyron equation for saturation pressure:
                
                $$P_{sat} = P_{ref} \\cdot \\exp\\left(\\frac{\\Delta H_{vap}}{R}\\left(\\frac{1}{T_{ref}} - \\frac{1}{T}\\right)\\right)$$
                """)
                
                T_K = amb_temp_C + 273.15
                cc_const = DELTA_H_VAP / R_GAS
                cc_offset = cc_const / T_REF
                p_sat_calc = P_REF * math.exp(cc_offset - cc_const / T_K)
                
                st.latex(f"P_{{sat}} = {P_REF} \\cdot \\exp\\left(\\frac{{{DELTA_H_VAP:.0f}}}{{{R_GAS:.3f}}}\\left(\\frac{{1}}{{{T_REF:.2f}}} - \\frac{{1}}{{{T_K:.2f}}}\\right)\\right) = {p_sat_calc:.2f} \\text{{ Pa}}")
                
                st.markdown("Water vapor mole fraction:")
                st.latex(f"X_{{H_2O}} = \\frac{{RH \\cdot P_{{sat}}}}{{P_{{amb}}}} = \\frac{{{rel_humid_pct:.1f}\\% \\cdot {p_sat_calc:.2f}}}{{{amb_pressure_Pa:.0f}}} = {X_H2O_initial:.6f}")
                
                st.markdown("---")
                st.markdown("#### 2. Ambient Oxygen Mole Fraction")
                st.latex(f"X_{{O_2}}^{{amb}} = (1 - X_{{H_2O}}) \\cdot X_{{O_2}}^0 = (1 - {X_H2O_initial:.6f}) \\cdot {X_O2_0:.6f} = {X_O2_amb:.6f}")
                
                st.markdown("---")
                st.markdown("#### 3. Oxygen Depletion Factor (φ)")
                st.markdown("""
                $$\\phi = \\frac{X_{O_2}^0 (1 - X_{CO_2} - X_{CO}) - X_{O_2}(1 - X_{CO_2}^0)}{X_{O_2}^0 (1 - X_{CO_2} - X_{CO} - X_{O_2})}$$
                """)
                
                phi_num = X_O2_0 * (1 - X_CO2_cal - X_CO_cal) - X_O2_cal * (1 - X_CO2_0)
                phi_den = X_O2_0 * (1 - X_CO2_cal - X_CO_cal - X_O2_cal)
                
                st.latex(f"\\phi = \\frac{{{X_O2_0:.6f} \\cdot (1 - {X_CO2_cal:.6f} - {X_CO_cal:.8f}) - {X_O2_cal:.6f} \\cdot (1 - {X_CO2_0:.6f})}}{{{X_O2_0:.6f} \\cdot (1 - {X_CO2_cal:.6f} - {X_CO_cal:.8f} - {X_O2_cal:.6f})}}")
                st.latex(f"\\phi = \\frac{{{phi_num:.8f}}}{{{phi_den:.8f}}} = {phi_iso:.6f}")
                
                st.markdown("---")
                st.markdown("#### 4. Heat Release Rate from Methane")
                st.markdown("""
                $$\\dot{q}_{CH_4} = \\dot{m}_{CH_4} \\cdot \\Delta H_c^{CH_4}$$
                """)
                st.latex(f"\\dot{{q}}_{{CH_4}} = {m_dot_CH4_kg_s:.6e} \\text{{ kg/s}} \\cdot {DELTA_HC_METHANE} \\text{{ MJ/kg}} = {q_dot_CH4_MW:.6f} \\text{{ MW}}")
                
                st.markdown("---")
                st.markdown("#### 5. C-Factor Calculation")
                st.markdown("""
                Rearranging the HRR equation:
                
                $$C = \\frac{\\dot{q}}{1.10 \\cdot E \\cdot X_{O_2}^{amb} \\cdot \\sqrt{\\frac{\\Delta P}{T_e}} \\cdot \\frac{\\phi - 0.172(1-\\phi)\\frac{X_{CO}}{X_{O_2}}}{1 - \\phi + 1.105\\phi}}$$
                """)
                
                # Calculate intermediate terms
                flow_term = sqrt(delta_P_cal / T_e_cal)
                co_correction = 0.172 * (1 - phi_iso) * (X_CO_cal / X_O2_cal) if X_O2_cal > 0 else 0
                bracket_num = phi_iso - co_correction
                bracket_den = 1 - phi_iso + 1.105 * phi_iso
                bracket_term = bracket_num / bracket_den
                
                st.markdown("**Intermediate calculations:**")
                st.latex(f"\\sqrt{{\\frac{{\\Delta P}}{{T_e}}}} = \\sqrt{{\\frac{{{delta_P_cal:.2f}}}{{{T_e_cal:.1f}}}}} = {flow_term:.6f}")
                st.latex(f"\\text{{CO correction}} = 0.172 \\cdot (1 - {phi_iso:.6f}) \\cdot \\frac{{{X_CO_cal:.8f}}}{{{X_O2_cal:.6f}}} = {co_correction:.8f}")
                st.latex(f"\\text{{Bracket term}} = \\frac{{{phi_iso:.6f} - {co_correction:.8f}}}{{1 - {phi_iso:.6f} + 1.105 \\cdot {phi_iso:.6f}}} = \\frac{{{bracket_num:.6f}}}{{{bracket_den:.6f}}} = {bracket_term:.6f}")
                
                st.markdown("**Final calculation:**")
                denominator = 1.10 * E_value * X_O2_amb * flow_term * bracket_term
                st.latex(f"C = \\frac{{{q_dot_CH4_MW:.6f}}}{{1.10 \\cdot {E_value} \\cdot {X_O2_amb:.6f} \\cdot {flow_term:.6f} \\cdot {bracket_term:.6f}}}")
                st.latex(f"C = \\frac{{{q_dot_CH4_MW:.6f}}}{{{denominator:.6f}}} = \\boxed{{{c_factor_iso:.6f}}}")
                
                st.markdown("---")
                st.markdown("#### Summary of Values Used")
                
                summary_col1, summary_col2 = st.columns(2)
                with summary_col1:
                    st.markdown("**Baseline Values:**")
                    st.text(f"X_O2_0 = {X_O2_0:.6f}")
                    st.text(f"X_CO2_0 = {X_CO2_0:.6f}")
                    
                    st.markdown("**Calibration Values:**")
                    st.text(f"X_O2 = {X_O2_cal:.6f}")
                    st.text(f"X_CO2 = {X_CO2_cal:.6f}")
                    st.text(f"X_CO = {X_CO_cal:.8f}")
                    st.text(f"T_e = {T_e_cal:.1f} K")
                    st.text(f"ΔP = {delta_P_cal:.2f} Pa")
                
                with summary_col2:
                    st.markdown("**Ambient Conditions:**")
                    st.text(f"T_amb = {amb_temp_C:.1f} °C")
                    st.text(f"P_amb = {amb_pressure_Pa:.0f} Pa")
                    st.text(f"RH = {rel_humid_pct:.1f} %")
                    st.text(f"P_sat = {p_sat_calc:.2f} Pa")
                    st.text(f"X_H2O = {X_H2O_initial:.6f}")
                    st.text(f"X_O2_amb = {X_O2_amb:.6f}")
                    
                    st.markdown("**Constants:**")
                    st.text(f"E = {E_value} MJ/kg")
                    st.text(f"ΔHc_CH4 = {DELTA_HC_METHANE} MJ/kg")
            
            st.divider()

st.markdown("#### Notes")
readme = SCRIPT_DIR / "README.md"
section_title = "### CFactor Check"

try:
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
except FileNotFoundError:
    st.caption("README.md not found")
except Exception as e:
    st.caption(f"Could not load notes: {e}")