import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import make_interp_spline
import matplotlib.cm as cm
import matplotlib.colors as mcolors

# Read the Excel file
file_path = r"ConeReport.xlsx"
sheet_name1 = 'Sheet2'
sheet_name2 = 'Sheet5'
sheet_name3 = 'Sheet7'

df1 = pd.read_excel(file_path, sheet_name=sheet_name1)
df2 = pd.read_excel(file_path, sheet_name=sheet_name2)
df3 = pd.read_excel(file_path, sheet_name=sheet_name3)

# Display the first few rows to understand the structure
print("Sheet 2:")
print(df1.head())
print(df1.columns)
print("\nSheet 5:")
print(df2.head())
print(df2.columns)
print("\nSheet 7:")
print(df3.head())
print(df3.columns)

# Define colors for each wood type (Plot 1 and Plot 3)
colors = {
    'Balsa': '#1f77b4',
    'Red Cedar': "#e7200a",
    'Pine': '#2ca02c',
    'Fir': "#d6951c",
    'Cypress': '#9467bd',
    'Redwood': '#8c564b',
    'Purple Heart': '#e377c2',
    'Ebony': '#7f7f7f',
    'Ipe': '#bcbd22'
}

# Conversion factor for Sheet 7 data (kW to kW/m²)
SAMPLE_AREA = 0.00884  # m²

# ============== FIGURE 1: Different Wood Types ==============
fig1, ax1 = plt.subplots(figsize=(14, 8))

time1 = df1.iloc[:, 0]

for i, (wood, color) in enumerate(colors.items()):
    col_idx = i + 1
    
    if col_idx < len(df1.columns):
        hrr = df1.iloc[:, col_idx]
        
        mask = ~(hrr.isna() | time1.isna())
        time_clean = time1[mask].values
        hrr_clean = hrr[mask].values
        
        if len(time_clean) > 3:  # Need at least 4 points for cubic spline
            spl = make_interp_spline(time_clean, hrr_clean, k=3)
            time_smooth = np.linspace(time_clean.min(), time_clean.max(), 300)
            hrr_smooth = spl(time_smooth)
            
            ax1.plot(time_smooth, hrr_smooth, label=wood, color=color, linewidth=2.5)

ax1.set_xlabel('Time (s)', fontsize=16)
ax1.set_ylabel('HRR (kW/m²)', fontsize=16)
ax1.set_title('HRR by Wood Type', fontsize=18)
ax1.legend(loc='best', fontsize=14)
ax1.grid(False)
ax1.tick_params(axis='both', labelsize=12)

fig1.tight_layout()

# ============== FIGURE 2: Red Cedar at Different Heat Fluxes ==============
fig2, ax2 = plt.subplots(figsize=(14, 8))

time2 = df2.iloc[:, 0]

# Get column names (assuming they contain heat flux values)
heat_flux_columns = df2.columns[1:]  # All columns except time
num_curves = len(heat_flux_columns)

# Extract heat flux values from column names
heat_fluxes = []
for col in heat_flux_columns:
    try:
        flux_value = float(''.join(filter(lambda x: x.isdigit() or x == '.', str(col))))
        heat_fluxes.append(flux_value)
    except:
        heat_fluxes.append(0)

# Create colormap (blue to red)
cmap = cm.coolwarm
norm = mcolors.Normalize(vmin=min(heat_fluxes), vmax=max(heat_fluxes))

for i, col in enumerate(heat_flux_columns):
    hrr = df2[col]
    
    mask = ~(hrr.isna() | time2.isna())
    time_clean = time2[mask].values
    hrr_clean = hrr[mask].values
    
    if len(time_clean) > 3:
        spl = make_interp_spline(time_clean, hrr_clean, k=3)
        time_smooth = np.linspace(time_clean.min(), time_clean.max(), 300)
        hrr_smooth = spl(time_smooth)
        
        color = cmap(norm(heat_fluxes[i]))
        ax2.plot(time_smooth, hrr_smooth, label=f'{heat_fluxes[i]:.0f} kW/m²', 
                 color=color, linewidth=2.5)

ax2.set_xlabel('Time (s)', fontsize=16)
ax2.set_ylabel('HRR (kW/m²)', fontsize=16)
ax2.legend(loc='best', fontsize=14, title='Heat Flux', title_fontsize=14)
ax2.grid(False)
ax2.tick_params(axis='both', labelsize=12)

# Add colorbar for heat flux
sm = cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cbar = fig2.colorbar(sm, ax=ax2, pad=0.02)
cbar.set_label('Heat Flux (kW/m²)', fontsize=16)
cbar.ax.tick_params(labelsize=12)

fig2.tight_layout()

# ============== FIGURE 3: Curve Clouds by Species ==============
fig3, ax3 = plt.subplots(figsize=(14, 8))

time3 = df3.iloc[:, 0]

# Group columns by species name
species_columns = {}
for col in df3.columns[1:]:  # Skip time column
    species_name = str(col).strip()
    # Handle cases where column names might have suffixes like "Balsa_1", "Balsa_2", etc.
    # Extract base species name
    base_name = None
    for species in colors.keys():
        if species.lower() in species_name.lower():
            base_name = species
            break
    
    if base_name:
        if base_name not in species_columns:
            species_columns[base_name] = []
        species_columns[base_name].append(col)

print("\nSpecies columns found:")
for species, cols in species_columns.items():
    print(f"  {species}: {len(cols)} columns")

# Create common time array for interpolation
time_common = np.linspace(time3.min(), time3.max(), 300)

# Plot curve clouds for each species
for species, cols in species_columns.items():
    if species not in colors:
        continue
    
    color = colors[species]
    all_hrr_smooth = []
    
    # Process each individual curve for this species
    for col in cols:
        hrr = df3[col]
        
        mask = ~(hrr.isna() | time3.isna())
        time_clean = time3[mask].values
        hrr_clean = hrr[mask].values
        
        # Convert from kW to kW/m²
        hrr_clean = hrr_clean / SAMPLE_AREA
        
        if len(time_clean) > 3:
            try:
                spl = make_interp_spline(time_clean, hrr_clean, k=3)
                # Interpolate only within the valid range of this curve
                valid_time = time_common[(time_common >= time_clean.min()) & 
                                         (time_common <= time_clean.max())]
                hrr_smooth = spl(valid_time)
                all_hrr_smooth.append((valid_time, hrr_smooth))
            except:
                continue
    
    if len(all_hrr_smooth) > 0:
        # Find common time range across all curves for this species
        min_time = max([arr[0].min() for arr in all_hrr_smooth])
        max_time = min([arr[0].max() for arr in all_hrr_smooth])
        
        # Create common time grid
        common_time = np.linspace(min_time, max_time, 300)
        
        # Interpolate all curves to common time grid
        hrr_matrix = []
        for valid_time, hrr_smooth in all_hrr_smooth:
            try:
                hrr_interp = np.interp(common_time, valid_time, hrr_smooth)
                hrr_matrix.append(hrr_interp)
            except:
                continue
        
        if len(hrr_matrix) > 0:
            hrr_matrix = np.array(hrr_matrix)
            
            # Calculate statistics for the cloud
            hrr_min = np.min(hrr_matrix, axis=0)
            hrr_max = np.max(hrr_matrix, axis=0)
            hrr_mean = np.mean(hrr_matrix, axis=0)
            
            # Plot the shaded region (cloud)
            ax3.fill_between(common_time, hrr_min, hrr_max, 
                            color=color, alpha=0.3, label=f'{species} (range)')
            
            # Plot the mean curve
            ax3.plot(common_time, hrr_mean, color=color, linewidth=2.5, 
                    label=f'{species} (mean)')

ax3.set_xlabel('Time (s)', fontsize=16)
ax3.set_ylabel('HRR (kW/m²)', fontsize=16)
#ax3.set_title('HRR Curve Clouds by Wood Species', fontsize=18)
ax3.legend(loc='best', fontsize=10, ncol=2)
ax3.grid(False)
ax3.tick_params(axis='both', labelsize=12)

fig3.tight_layout()

# ============== FIGURE 4: THR vs Incident Energy ==============
fig4, ax4 = plt.subplots(figsize=(14, 8))

HEAT_FLUX = 50  # kW/m²

# Plot THR clouds for each species
for species, cols in species_columns.items():
    if species not in colors:
        continue
    
    color = colors[species]
    all_thr_smooth = []
    all_e0_smooth = []
    
    # Process each individual curve for this species
    for col in cols:
        hrr = df3[col]
        
        mask = ~(hrr.isna() | time3.isna())
        time_clean = time3[mask].values
        hrr_clean = hrr[mask].values
        
        # Convert from kW to kW/m²
        hrr_clean = hrr_clean / SAMPLE_AREA
        
        # Calculate incident energy E_0 (MJ/m²)
        e0 = time_clean * HEAT_FLUX / 1000  # kJ/m² -> MJ/m²
        
        # Calculate THR (MJ/m²)
        # HRR is in kW/m², divide by 1000 to get MW/m²
        # Then cumulative sum gives MJ/m² (since time step is 1 second)
        dt = np.diff(time_clean, prepend=time_clean[0])
        dt[0] = time_clean[1] - time_clean[0] if len(time_clean) > 1 else 1
        thr = np.cumsum(hrr_clean / 1000 * dt)  # MJ/m²
        
        if len(e0) > 3:
            try:
                spl = make_interp_spline(e0, thr, k=3)
                e0_smooth = np.linspace(e0.min(), e0.max(), 300)
                thr_smooth = spl(e0_smooth)
                all_e0_smooth.append(e0_smooth)
                all_thr_smooth.append(thr_smooth)
            except:
                continue
    
    if len(all_thr_smooth) > 0:
        # Find common E0 range across all curves for this species
        min_e0 = max([arr.min() for arr in all_e0_smooth])
        max_e0 = min([arr.max() for arr in all_e0_smooth])
        
        # Create common E0 grid
        common_e0 = np.linspace(min_e0, max_e0, 300)
        
        # Interpolate all curves to common E0 grid
        thr_matrix = []
        for e0_smooth, thr_smooth in zip(all_e0_smooth, all_thr_smooth):
            try:
                thr_interp = np.interp(common_e0, e0_smooth, thr_smooth)
                thr_matrix.append(thr_interp)
            except:
                continue
        
        if len(thr_matrix) > 0:
            thr_matrix = np.array(thr_matrix)
            
            # Calculate statistics for the cloud
            thr_min = np.min(thr_matrix, axis=0)
            thr_max = np.max(thr_matrix, axis=0)
            thr_mean = np.mean(thr_matrix, axis=0)
            
            # Plot the shaded region (cloud)
            ax4.fill_between(common_e0, thr_min, thr_max, 
                            color=color, alpha=0.3, label=f'{species} (range)')
            
            # Plot the mean curve
            ax4.plot(common_e0, thr_mean, color=color, linewidth=2.5, 
                    label=f'{species} (mean)')

ax4.set_xlabel('E₀ (MJ/m²)', fontsize=16)
ax4.set_ylabel('THR (MJ/m²)', fontsize=16)
#ax4.set_title('THR vs Incident Energy by Wood Species', fontsize=18)
ax4.legend(loc='best', fontsize=10, ncol=2)
ax4.grid(False)
ax4.tick_params(axis='both', labelsize=12)

fig4.tight_layout()

# Save figures
fig2.savefig('HRR_Hor_RC.png', dpi=300, bbox_inches='tight')
fig3.savefig('HRR.png', dpi=300, bbox_inches='tight')
fig4.savefig('THR.png', dpi=300, bbox_inches='tight')

plt.show()