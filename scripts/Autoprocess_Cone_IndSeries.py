#!/usr/bin/env python3
"""

Script to average replicates of Cone data within a series

"""

import datetime
import json
import logging
import math
import os
import re
from pathlib import Path
from collections import OrderedDict

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter, find_peaks

DEBUG = False
logging.basicConfig(level=logging.DEBUG if DEBUG else logging.INFO)
logger = logging.getLogger(__name__)
import warnings

import avg_tools_PH as ph
from utils import colorize

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=pd.errors.SettingWithCopyWarning)


def average_cone_series(series_name: str, data_dir: Path, metadata_dir: Path, matl_dir: Path):                                                    

    paths = list(data_dir.rglob(f"{series_name}_[rR]*.csv"))
    print(paths)
    badpaths = []
    Flux = int(re.search(r'(\d+)', str(series_name).split('_')[1]).group(1))
    Folder = paths[0].parent
    Dataframes_Whole = []
    Dataframes_HRRPUA = [] 
    masses = []
    rep =  []
    surf_areas = []
    res_masses = []
    X_CO2_initials =[]
    X_CO_initials =[]
    X_O2_initials =[]
    ambient_temperatures = []
    ambient_pressures = []
    humidities = []
    ignition_times = []
    flameout_times = []
    for path in paths:
        filename = path.name
        match = re.search(r'_R(\d+)\.csv$', filename)
        replicate_number = match.group(1)  # Get the matched number of replicate
        file_meta = next(metadata_dir.rglob(str(Path(filename).with_suffix('.json'))),None)
        if not file_meta:
            print(colorize(f"Skipping {filename}, no metadata file found","red")) ####### Safeguard on nometadata
            badpaths.append(path)
            continue
        with open(file_meta, "r") as r: #Open json metadata file and extract sample mass
            data = json.load(r)
        if data["Pass Review"] == False:
            print(colorize(f"Skipping {filename}, failed manual review","red")) ####### Remove all tests that failed manual review
            badpaths.append(path)
            continue
        elif data["Pass Review"] == True:
            logger.info(f"Reautoprocessing {filename}, passed manual review")
        else:
            logger.info(f"Autprocessing {filename}") 
        sample_mass = float(data.get("Sample Mass (g)")) if data.get("Sample Mass (g)") else None
        res_mass = float(data.get('Residual Mass (g)')) if data.get('Residual Mass (g)') else None
        sa = float(data.get('Surface Area (m2)'))if data.get('Surface Area (m2)') else None
        X_O2_i = data.get("X_O2 Initial")
        X_CO2_i = data.get("X_CO2 Initial")
        X_CO_i = data.get("X_CO Initial")
        rel_humid = data.get('Relative Humidity (%)')
        amb_temp = data.get("Ambient Temperature (°C)")
        amb_pressure = data.get("Barometric Pressure (Pa)")
        rep.append(replicate_number)  # Append to the list
        masses.append(sample_mass)
        surf_areas.append(sa)
        res_masses.append(res_mass)
        humidities.append(rel_humid)
        ambient_pressures.append(amb_pressure)
        ambient_temperatures.append(amb_temp)
        X_CO2_initials.append(X_CO2_i)
        X_CO_initials.append(X_CO_i)
        X_O2_initials.append(X_O2_i)
        ignition_times.append(int(data.get('t_ignition (s)')) if data.get('t_ignition (s)') else None)
        flameout_times.append(int(data.get('t_flameout (s)')) if data.get('t_flameout (s)') else None)
    # Remove all tests that failed manual review from the list of paths
    for path in badpaths:   
        paths.remove(path)
    if len(paths) == 0:
        raise Exception(colorize(f"No files found for series {series_name}", "red"))
       #------------------------------------------------------  
    # Read all the good data 
    #------------------------------------------------------ 
    # Read data
    
    for i, path in enumerate(paths):
        sample_mass = masses[i]
        res_mass = res_masses[i]
        sa = surf_areas[i]
        X_O2_i = X_O2_initials[i]
        X_CO2_i = X_CO2_initials[i]
        X_CO_i = X_CO_initials[i]
        rel_humid = humidities[i]
        amb_temp = ambient_temperatures[i]
        amb_pressure = ambient_pressures[i]
        df = pd.read_csv(path).dropna(how='all').reset_index(drop=True)
        df['Time (s)'] = df['Time (s)'].astype(float)
        df['dt'] = df["Time (s)"].diff()
        df['t*EHF (MJ/m2)'] = (df['Time (s)'] * Flux)/1000

        
        #Mass and MLR calculations
        if not df["Mass (g)"].isnull().all():
            m = df["Mass (g)"]
            for j in range(len(df)):
                if j == 0:
                    df.loc[j,"MLR (g/s)"] = (25*m[0] - 48*m[1] + 36*m[2] - 16*m[3] + 3*m[4])/(12*df['dt'].iloc[j])
                elif j == 1:
                    df.loc[j,"MLR (g/s)"] = (3*m[0] + 10*m[1] - 18*m[2] + 6*m[3] - m[4])/(12*df['dt'].iloc[j])
                elif j == len(df)-2:
                    df.loc[j,"MLR (g/s)"] = (-3*m[j+1] - 10*m[j] + 18*m[j-1] - 6*m[j-2] + m[j-3])/(12*df['dt'].iloc[j])
                elif j == len(df)-1:
                    df.loc[j,"MLR (g/s)"] = (-25*m[j] + 48*m[j-1] - 36*m[j-2] + 16*m[j-3] - 3*m[j-4])/(12*df['dt'].iloc[j])
                else:
                    df.loc[j,"MLR (g/s)"] = (-m[j-2]+ 8*m[j-1]- 8*m[j+1]+m[j+2])/(12*df['dt'].iloc[j])
                    
            df['Mass Loss (g)'] = sample_mass - df['Mass (g)']
            df['Mass LossPUA (g/m2)'] = df['Mass Loss (g)'] / sa if sa else None
            df['MLRPUA (g/s-m2)'] = df['MLR (g/s)'] / sa if sa else None
        elif "MLR (g/s)" in df.columns:
            df['Mass Loss (g)'] = (
                pd.to_numeric(df['MLR (g/s)'], errors='coerce')
                .where(lambda x: np.isfinite(x), 0) *
                pd.to_numeric(df['dt'], errors='coerce')
                .where(lambda x: np.isfinite(x), 0)
            ).cumsum() #USE THIS ONLY TO FIND THE 10% AND 90% MASS LOSS TIMES
            df['MLRPUA (g/s-m2)'] = df['MLR (g/s)'] / sa if sa else None
        elif "MLRPUA (g/s-m2)" in df.columns:
            df['Mass LossPUA (g/m2)'] = (
                pd.to_numeric(df['MLRPUA (g/s-m2)'], errors='coerce')
                .where(lambda x: np.isfinite(x), 0) *
                pd.to_numeric(df['dt'], errors='coerce')
                .where(lambda x: np.isfinite(x), 0)
            ).cumsum() #USE THIS ONLY TO FIND THE 10% AND 90% MASS LOSS TIMES
            df['Mass Loss (g)'] = df['Mass LossPUA (g/m2)'] * sa if sa else None
            df['MLR (g/s)'] = df['MLRPUA (g/s-m2)'] * sa if sa else None
        
        if df['Mass (g)'].iloc[0] < 0: #Handle FTT cases with bad mass data
            df['Mass (g)'] = None
            
        if df['Mass (g)'].isnull().all(): #Handle cases with no mass data
            df['Mass Loss (g)'] = None
            df['Mass LossPUA (g/m2)'] = None
            df['MLRPUA (g/s-m2)'] =  None
        


        #Heat Release Calculations
        if "HRRPUA (kW/m2)" not in df.columns:
            df["HRRPUA (kW/m2)"] = df["HRR (kW)"] / sa if sa is not None else None
        else:
            df["HRR (kW)"] = df["HRRPUA (kW/m2)"] * sa if sa is not None else None
        df['Q (MJ)'] = (df['HRR (kW)'] * df['dt']) / 1000
        df['QPUA (MJ/m2)'] = (df['HRRPUA (kW/m2)'] * df['dt']) / 1000
        df['Combustability'] = (df['QPUA (MJ/m2)'].cumsum()/df['t*EHF (MJ/m2)'])


        #Smoke and Gas Production Calculations
        #Needed for calculating yields/extinction coeff, have to either carry this or
        #weight air taken from 2077, this publication also used ambient pressure in the building, so will I
        W_air = 28.97
        df['Rho_Air (kg/m3)'] = ((amb_pressure/1000) * W_air) / (8.314 * df['T Duct (K)']) if amb_pressure else None
        # Volumetric flow rate (m^3/s)
        df["V Duct (m3/s)"]   = df['MFR (kg/s)'] / df["Rho_Air (kg/m3)"]


        #Finding Soot  production based on FCD User Guide- but bring area into eq so have Vduct
        #Says to use smoke production sigmas = 8.7m2/g, not sigmaf
        df['Soot Production (g/s)'] = 1/8.7 * df["K Smoke (1/m)"] * df['V Duct (m3/s)']
        df['Smoke Production (m2/s)'] = df['K Smoke (1/m)'] * df['V Duct (m3/s)']
        ## Gas Production and Yield

        W_CO2 = 44.01
        W_CO = 28.01
        W_O2 = 32
        
        if X_O2_i:
            #Production and yields calculated by following FCD user guide
            df['CO2 Production (g/s)'] = (W_CO2/W_air) * (df['CO2 (Vol fr)'] - X_CO2_initials[i]) * df['MFR (kg/s)'] * 1000
            df['CO Production (g/s)'] = (W_CO/W_air) * (df['CO (Vol fr)'] - X_CO_initials[i]) * df['MFR (kg/s)'] * 1000
            df['O2 Consumption (g/s)'] = (W_O2/W_air) * (X_O2_initials[i] - df['O2 (Vol fr)']) * df['MFR (kg/s)'] * 1000
            df['Total O2 (g)'] = (df['O2 Consumption (g/s)'] * df['dt']).cumsum()
            df['Total CO2 (g)'] = (df['CO2 Production (g/s)'] * df['dt']).cumsum()
            df['Total CO (g)'] = (df['CO Production (g/s)'] * df['dt']).cumsum()
            df['Total Soot (g)'] = (df['Soot Production (g/s)'] * df['dt']).cumsum()
        


        Dataframes_Whole.append(df)
        df_HRRPUA = df[['Time (s)', 'HRRPUA (kW/m2)']].copy()
        Dataframes_HRRPUA.append(df_HRRPUA)

    #Merge HRR curves for whole curve statistics, rest will just be key features.

    merged_HRRPUA = Dataframes_HRRPUA[0].copy()
    for data in Dataframes_HRRPUA[1:]:
        merged_HRRPUA = pd.merge(merged_HRRPUA, data, on='Time (s)', how='outer', suffixes=('', f' {len(merged_HRRPUA.columns)}'))
    merged_HRRPUA.rename(columns={'HRRPUA (kW/m2)': 'HRRPUA (kW/m2) 1'}, inplace=True)

    # Number of good experiments
    N_exp = len(merged_HRRPUA.columns)-1
    #------------------------------------------------------  
    # Average/STD
    #------------------------------------------------------ 
    # Compute the average HRR, MLR for the series, ignoring NaN values
    Average_HRRPUA = merged_HRRPUA.iloc[:, 1:].mean(axis=1, skipna=True)
    Std_HRRPUA = merged_HRRPUA.iloc[:, 1:].std(axis=1, skipna=True)
    Stdm_HRRPUA = Std_HRRPUA / math.sqrt(N_exp)

    # Need to add more parameters
    Unc_HRRPUA = np.sqrt((Std_HRRPUA/Average_HRRPUA)**2) * Average_HRRPUA
    
    # Average df
    Average_df = pd.DataFrame({'Time (s)': merged_HRRPUA['Time (s)'],
                               'HRRPUA (kW/m2)': Average_HRRPUA,
                              'Uc HRRPUA (kW/m2)': Unc_HRRPUA})
    
    #------------------------------------------------------  
    # Statistics and outliers
    #------------------------------------------------------ 
    if N_exp >= 3:
        print(colorize(f"{N_exp} good tests found, performing statistical analysis", "green"))
        HRR_out = ph.outlier(merged_HRRPUA, Average_HRRPUA, Std_HRRPUA)
        Output_file = str(Folder /series_name)+'_Average.csv'
        Average_df.to_csv(Output_file, index=False)
        print(colorize(f"Averaged data saved to: {Output_file}", "green"))
    else:
        print(colorize("Not enough tests to perform statistical analysis, skipping outlier detection", "red"))
        HRR_out = [None] *len(paths)
        Output_file = str(Folder /series_name)+'_Average.csv'
        Average_df[:] = np.nan  # Set all values in the DataFrame to NaN
        Average_df.to_csv(Output_file, index=False)
        print(colorize(f"Average file {Output_file} remains blank until more tests are performed", "red"))
    
#------------------------------------------------------  
    # Key value calculations
    #------------------------------------------------------ 
    # Make dataframe to store all values 
    columns = ['t_ign', 'm_ign', 'r_yield', 'HRR_60', 'HRR_180', 'HRR_300', 'stdy_MLR', 'peak_MLR', 'stdy_HRR', "peak_HRR", 'THR'
               ,'HoC','ext_area', 'pre_smoke', 'post_smoke', 'tot_smoke', "Y_soot", 'Y_CO2', 'Y_CO', 'FGP', 'E_ign', 't_out']
    df_values = pd.DataFrame(columns=columns)


    for i, path in enumerate(paths):
        #Pull Dataframes/values
        df_tot = Dataframes_Whole[i]
        sample_mass = masses[i]
        res_mass = res_masses[i]
        sa = surf_areas[i]
        X_O2_i = X_O2_initials[i]
        X_CO2_i = X_CO2_initials[i]
        X_CO_i = X_CO_initials[i]
        rel_humid = humidities[i]
        amb_temp = ambient_temperatures[i]
        amb_pressure = ambient_pressures[i]
        
        ###Bounds Set Up
        peak_HRR = df_tot['HRRPUA (kW/m2)'].max()
        peak_idx = df_tot['HRRPUA (kW/m2)'].idxmax()
        peak_MLR = df_tot['MLRPUA (g/s-m2)'].max() if 'MLRPUA (g/s-m2)' in df_tot.columns else None
        t_ign = ignition_times[i]
        t_out = flameout_times[i]
        i_ign = (
            abs(df_tot['Time (s)'] - t_ign).idxmin()
            if t_ign is not None
            else None
        )
        i_out = (
            abs(df_tot['Time (s)'] - t_out).idxmin()
            if t_out is not None
            else None
        )
        totalmassloss = sample_mass - res_mass if sample_mass and res_mass else None
        m_ign = df_tot['Mass (g)'].iloc[i_ign] if i_ign is not None else None
        r_yield = (res_mass/sample_mass) * 100 if res_mass and sample_mass else None
        if df_tot['Mass Loss (g)'].isnull().all() and not df_tot['Mass LossPUA (g/m2)'].isnull().all():
            ml_10 = .1 * (df_tot['Mass LossPUA (g/m2)'].iloc[-1])
            ml_90 = .9 * (df_tot['Mass LossPUA (g/m2)'].iloc[-1])
            t_10 = (df_tot['Mass LossPUA (g/m2)'] - ml_10).abs().idxmin()
            t_90 = (df_tot['Mass LossPUA (g/m2)'] - ml_90).abs().idxmin()
        elif not df_tot['Mass Loss (g)'].isnull().all():
            ml_10 = .1 * (df_tot['Mass Loss (g)'].iloc[-1])
            ml_90 = .9 * (df_tot['Mass Loss (g)'].iloc[-1])
            t_10 = (df_tot['Mass Loss (g)'] - ml_10).abs().idxmin()
            t_90 = (df_tot['Mass Loss (g)'] - ml_90).abs().idxmin()
        else: #Fallback if no mass data, integrate by total heat release
            thr_10 = .1 * (df_tot['QPUA (MJ/m2)'].iloc[-1])
            thr_90 = .9 * (df_tot['QPUA (MJ/m2)'].iloc[-1])
            t_10 = (df_tot['QPUA (MJ/m2)'] - thr_10).abs().idxmin()
            t_90 = (df_tot['QPUA (MJ/m2)'] - thr_90).abs().idxmin()
        
        i_10 = df_tot.index.get_loc(t_10)
        i_90 = df_tot.index.get_loc(t_90)
        m_10 = df_tot['Mass (g)'].iloc[i_10]
        m_90 = df_tot['Mass (g)'].iloc[i_90]
        test_length = df_tot['Time (s)'].iloc[-1]
        hrr_negatives = df_tot['HRRPUA (kW/m2)'].iloc[:peak_idx]
        hrr_negatives = hrr_negatives[hrr_negatives < 0]
        if len(hrr_negatives) > 0:
            i_afterneg = hrr_negatives.index[-1]+1
            t_afterneg = df_tot["Time (s)"].iloc[i_afterneg]
        else:
            i_afterneg = 0 #If not negative HRR, integrate the whole thing
            t_afterneg = df_tot["Time (s)"].iloc[0]

        ##HRR/MLR Quantities
        
        if type(m_10) == float and type(m_90) == float and sa:
            stdy_MLR = ((m_10 - m_90)/(t_90 - t_10))* (1/sa) #calculated per ISO5660
        else:
            stdy_MLR = df_tot['MLRPUA (g/s-m2)'][i_10:i_90].mean() #Fallback to mean MLRPUA if no mass data
        
        stdy_HRR = (df_tot['dt'].iloc[i_10:i_90].mean()/(t_90 - t_10)) *(0.5*df_tot['HRRPUA (kW/m2)'][i_10] + df_tot['HRRPUA (kW/m2)'][i_10+1: i_90].sum() + 0.5*df_tot['HRRPUA (kW/m2)'][i_90]) 
        ## For HRRPUA over first x seconds from ignition if have, or pt after last neg HRR if do not by ASTM, do we want to note this
        ##Calculate HRR averages using trapezium, as per ISO and ASTM standatds
        if t_ign:
            low_bound = i_ign
            low_bound_time = t_ign
            sixty_sec = low_bound_time + 60
            oneeighty_sec = low_bound_time + 180
            threehundred_sec = low_bound_time + 300
            sixty_idx = abs(df_tot['Time (s)'] - sixty_sec).idxmin()
            oneeighty_idx = abs(df_tot['Time (s)'] - oneeighty_sec).idxmin()
            threehundred_idx = abs(df_tot['Time (s)'] - threehundred_sec).idxmin()
        else:
            low_bound = i_afterneg
            low_bound_time = t_afterneg
            sixty_sec = low_bound_time + 60
            oneeighty_sec = low_bound_time + 180
            threehundred_sec = low_bound_time + 300
            sixty_idx = abs(df_tot['Time (s)'] - sixty_sec).idxmin()
            oneeighty_idx = abs(df_tot['Time (s)'] - oneeighty_sec).idxmin()
            threehundred_idx = abs(df_tot['Time (s)'] - threehundred_sec).idxmin()
        if test_length >= low_bound_time + 300:
            HRR_60 = (
                df_tot['dt'].iloc[low_bound:sixty_idx].mean() / 60 *
                (
                    0.5 * df_tot['HRRPUA (kW/m2)'].iloc[low_bound]
                    + df_tot['HRRPUA (kW/m2)'].iloc[low_bound + 1 : sixty_idx].sum()
                    + 0.5 * df_tot['HRRPUA (kW/m2)'].iloc[sixty_idx]
                )
            )
            HRR_180 = (
                df_tot['dt'].iloc[low_bound:oneeighty_idx].mean() / 180 *
                (
                    0.5 * df_tot['HRRPUA (kW/m2)'].iloc[low_bound]
                    + df_tot['HRRPUA (kW/m2)'].iloc[low_bound + 1 : oneeighty_idx].sum()
                    + 0.5 * df_tot['HRRPUA (kW/m2)'].iloc[oneeighty_idx]
                )
            )
            HRR_300 = (
                df_tot['dt'].iloc[low_bound:threehundred_idx].mean() / 300 *
                (
                    0.5 * df_tot['HRRPUA (kW/m2)'].iloc[low_bound]
                    + df_tot['HRRPUA (kW/m2)'].iloc[low_bound + 1 : threehundred_idx].sum()
                    + 0.5 * df_tot['HRRPUA (kW/m2)'].iloc[threehundred_idx]
                )
            )
        elif test_length >= low_bound_time + 180:
            HRR_60 = (
                df_tot['dt'].iloc[low_bound:sixty_idx].mean() / 60 *
                (
                    0.5 * df_tot['HRRPUA (kW/m2)'].iloc[low_bound]
                    + df_tot['HRRPUA (kW/m2)'].iloc[low_bound + 1 : sixty_idx].sum()
                    + 0.5 * df_tot['HRRPUA (kW/m2)'].iloc[sixty_idx]
                )
            )
            HRR_180 = (
                df_tot['dt'].iloc[low_bound:oneeighty_idx].mean() / 180 *
                (
                    0.5 * df_tot['HRRPUA (kW/m2)'].iloc[low_bound]
                    + df_tot['HRRPUA (kW/m2)'].iloc[low_bound + 1 : oneeighty_idx].sum()
                    + 0.5 * df_tot['HRRPUA (kW/m2)'].iloc[oneeighty_idx]
                )
            )
            HRR_300 = None
        elif test_length >= low_bound_time + 60:
            HRR_60 = (
                df_tot['dt'].iloc[low_bound:sixty_idx].mean() / 60 *
                (
                    0.5 * df_tot['HRRPUA (kW/m2)'].iloc[low_bound]
                    + df_tot['HRRPUA (kW/m2)'].iloc[low_bound + 1 : sixty_idx].sum()
                    + 0.5 * df_tot['HRRPUA (kW/m2)'].iloc[sixty_idx]
                )
            )
            HRR_180 = None
            HRR_300 = None
        else:
            HRR_60 = None
            HRR_180 = None
            HRR_300 = None

        ###Further derived quantities
        THR = df_tot['QPUA (MJ/m2)'][i_afterneg:].sum()
        if totalmassloss:
            THR_total = df_tot['Q (MJ)'].sum()
            HoC = THR_total/(totalmassloss/1000) #MJ/kg
            ext_area = ((df_tot['V Duct (m3/s)'] * df['K Smoke (1/m)'] * df['dt']).sum())/((totalmassloss/1000))#m2/kg
        else:
            THR_total = None
            HoC = None
            ext_area = None
        if t_ign:
            E_ign_kJ = Flux * t_ign
            E_ign = E_ign_kJ / 1000 #MJ/m2 to the metadata
            #FGP combustability calculated as max slope THRPUA vs t*EHF curve, following Richard Lyon paper, m2/J
            FGP = (1/E_ign_kJ) * (1/1000) * (df_tot['Combustability'].max())
            pre_smoke =1/sa * (df_tot['Smoke Production (m2/s)'][:t_ign] * df_tot['dt'][:t_ign]).sum() if sa else None
            post_smoke =1/sa * (df_tot['Smoke Production (m2/s)'][t_ign:] * df_tot['dt'][t_ign:]).sum() if sa else None
            tot_smoke =1/sa * (df_tot['Smoke Production (m2/s)'] * df_tot['dt']).sum() if sa else None
        else:
            E_ign = None
            FGP = None
            pre_smoke =1/sa * (df_tot['Smoke Production (m2/s)'] * df_tot['dt']).sum() if sa else None
            post_smoke = None
            tot_smoke =1/sa * (df_tot['Smoke Production (m2/s)'] * df_tot['dt']).sum() if sa else None

        
        if 'CO2 Production (g/s)' in columns: #actual data can calculate
            Y_soot = ((df_tot['Soot Production (g/s)'] * df_tot['dt']).sum()) / sample_mass
            Y_CO2 = ((df_tot['CO2 Production (g/s)'] * df_tot['dt']).sum()) / sample_mass
            Y_CO = ((df_tot['CO Production (g/s)'] * df_tot['dt']).sum()) / sample_mass
        else: #Use the average for given CO2 kg/kg columns
            Y_soot = None
            Y_CO = df_tot['CO (kg/kg)'].mean() if 'CO (kg/kg)' in df_tot.columns else None
            Y_CO2 = df_tot['CO2 (kg/kg)'].mean() if 'CO2 (kg/kg)' in df_tot.columns else None

        df_values.loc[len(df_values)]=[t_ign, m_ign, r_yield, HRR_60, HRR_180, HRR_300, stdy_MLR, peak_MLR, stdy_HRR, peak_HRR, THR,
                                        HoC, ext_area, pre_smoke, post_smoke, tot_smoke, Y_soot, Y_CO2, Y_CO, FGP, E_ign, t_out]
    # Average and standard deviation of key values
    df_values.loc['mean'] = df_values.mean()
    df_values.loc['std'] = df_values[:-1].std()
    df_values.loc['outlier'] = [0] * df_values.shape[1]
    df_values.loc['uncertainty'] = [0] * df_values.shape[1]
    for key in columns:
        df_values[key]['outlier',]  = ph.parameter_outliers(df_values[key][:-4], df_values[key]['mean'], df_values[key]['std'])
        df_values[key]['uncertainty'] = np.sqrt((df_values[key]['std']/df_values[key]['mean'])**2)*df_values[key]['mean'] if df_values[key]['mean'] !=0 else 0
    # Match column names with metadata entries
    df_values = df_values.rename(columns={
                                't_ign': "t_ignition (s)",    
                                'm_ign': 'm_ignition (g)',
                                'r_yield': 'Residue Yield (%)',
                                'HRR_60': "Average HRRPUA 60s (kW/m2)",
                                'HRR_180': "Average HRRPUA 180s (kW/m2)",
                                'HRR_300': "Average HRRPUA 300s (kW/m2)",
                                'stdy_MLR': "Steady Burning MLRPUA (g/s-m2)",
                                'peak_MLR': "Peak MLRPUA (g/s-m2)",
                                'stdy_HRR': "Steady Burning HRRPUA (kW/m2)",
                                'peak_HRR': "Peak HRRPUA (kW/m2)",
                                'THR': "Total Heat Release (MJ/m2)",
                                'HoC': "Average HoC (MJ/kg)",
                                'ext_area': "Average Specific Extinction Area (m2/kg)",
                                'pre_smoke': "Smoke Production Pre-ignition (m2/m2)",
                                'post_smoke': "Smoke Production Post-ignition (m2/m2)",
                                'tot_smoke': "Smoke Production Total (m2/m2)",
                                'Y_soot': "Y_Soot (g/g)",
                                'Y_CO2': "Y_CO2 (g/g)",
                                'Y_CO': "Y_CO (g/g)",
                                'FGP': "Fire Growth Potential (m2/J)",
                                'E_ign': "Ignition Energy (MJ/m2)",
                                't_out': "t_flameout (s)"
                            })
    

    #------------------------------------------------------  
    # Update metadata
    #------------------------------------------------------ 
    # Update material metadata
    
    #Onset values seem to be getting incorrectly populated in here, but they are being found correctly in df_values
    mat_ID = series_name.split("_" , 1)[0]
    matl_meta = matl_dir / f"{mat_ID}.json"
    #if matl_meta.exists():
    if matl_meta.exists():
        with open(matl_meta, "r") as f:
            matl_data = json.load(f)
        matl_data.update({series_name:{}})

        if N_exp >=3:
            testlist = []
            for path in paths:
                testlist.append(path.stem)
            matl_data[series_name]["Experiments"] = testlist

            for key in df_values.columns:
                matl_data[series_name][key] = float(f"{df_values[key]['mean']:.4g}")
                unc_key = 'Uc ' + key
                matl_data[series_name][unc_key] = float(f"{df_values[key]['uncertainty']:.4g}")
            

        else: 
                print(colorize(f'Not enough experiments to generate mean values and uncertainty for {mat_ID}', "red"))
        with open(matl_meta, "w") as f:
            f.write(json.dumps(matl_data, indent=4))
        print(colorize(f"Material metadata updated for {mat_ID}", "green"))
    else:
        print(colorize(f"Material metadata does not exist for {mat_ID}", "red"))



    # Updata test metadata
  
    counter=0
    for path in paths:
        # Open test metadata
        filename = path.name
        print(filename)
        match = re.search(r'_R(\d+)\.csv$', filename)
        replicate_number = match.group(1)  # Get the matched number of replicate
        file_meta = next(metadata_dir.rglob(str(Path(filename).with_suffix('.json'))),None)
        with open(file_meta, "r") as r:
            metadata = json.load(r)

        # Fill in test metadata
        if N_exp >=3:
            for key in df_values.columns:
                metadata[key] = float(f"{df_values[key][counter]:.4g}")
                outlier_key = key.split(' (')[0]+ ' Outlier'
                metadata[outlier_key] = df_values[key]['outlier'][counter]
        else: 
            for key in df_values.columns:
                metadata[key] = float(f"{df_values[key][counter]:.4g}")
        
        metadata["Autoprocessed"] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(metadata["Autoprocessed"])
        if HRR_out[counter] == True:
            metadata['Heat Release Rate Outlier'] = True
        elif HRR_out[counter] == False:
            metadata['Heat Release Rate Outlier'] = False   
        
        # replace all NaN values with None (which turns into null when serialized) to fit JSON spec (and get rid of red underlines)
        metadata = {k: v if v == v else None for k, v in metadata.items()}
        
        print(file_meta)
        # Save test metadata
        with open(file_meta, "w") as w:
            json.dump(metadata, w, indent=4)    
        counter = counter+1


if __name__ == "__main__":
    import argparse
    
    # Use script location as base for default paths
    SCRIPT_DIR = Path(__file__).parent
    PROJECT_ROOT = SCRIPT_DIR.parent
    # Set up command-line arguments
    parser = argparse.ArgumentParser(description='Process Cone series data')
    parser.add_argument('series_name', help='Series name to process')
    parser.add_argument('--root', type=str, 
                        default=str(PROJECT_ROOT),
                        help='Project root directory')
    parser.add_argument('--prepared', type=str,
                        help='Prepared data directory (default: ROOT/Exp-Data_Prepared-Final)')
    parser.add_argument('--metadata', type=str,
                        help='Metadata directory (default: ROOT/Metadata/Prepared-Final)')
    parser.add_argument('--materials', type=str, 
                        help='Materials metadata directory (default: ROOT/Metadata/Materials)')
    
    args = parser.parse_args()
    
    # Set up directories from arguments or defaults based on root
    ROOT_DIR= Path(args.root)
    prepared_dir = Path(args.prepared) if args.prepared else ROOT_DIR / "Exp-Data_Prepared-Final" 
    metadata_dir = Path(args.metadata) if args.metadata else ROOT_DIR / "Metadata" / "Prepared-Final"
    matl_dir = Path(args.materials) if args.materials else ROOT_DIR / "Metadata" / "Materials"
    series_name = args.series_name

    data = average_cone_series(series_name, prepared_dir, metadata_dir, matl_dir)
   