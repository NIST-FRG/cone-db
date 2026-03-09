import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

# Data from the table
#woods = ['Balsa', 'Red Cedar', 'Pine', 'Fir', 'Cypress', 'Redwood', 'Purple Heart', 'Ebony', 'Ipe']
woods = ['Balsa', 'Red Cedar', 'Pine', 'Fir', 'Cypress', 'Redwood', 'Purple Heart', 'Ebony']



# Density data (g/cm³)
density = [0.15, 0.34, 0.38, 0.40, 0.43, 0.45, 0.83, 0.88, 0.96]

# TTI (Time to Ignition) data (s)
tti = [7.0, 18.6, 22.3, 20.7, 28.7, 26.7, 42.3, 53.0, 24.2]
tti_err = [1.0, 5.0, 2.7, 4.7, 1.5, 7.1, 4.9, 8.0, 17.3]

# First Peak HRR data (kW/m²)
first_peak_hrr = [183.9, 216.8, 226.0, 210.5, 183.8, 162.2, 187.9, 220.9, 221.0]
first_peak_hrr_err = [7.2, 36.9, 6.6, 13.7, 5.9, 41.2, 21.3, 46.0, 30.4]

# First Peak Time (s)
first_peak_time = [28.0, 42.0, 44.3, 45.0, 53.0, 51.3, 73.0, 78.3, 68.9]
first_peak_time_err = [2.4, 7.7, 5.6, 4.9, 1.6, 13.1, 5.9, 8.2, 11.5]

# Second Peak HRR data (kW/m²)
second_peak_hrr = [73.3, 124.3, 142.7, 177.5, 171.6, 139.0, 241.6, 292.2, 410.2]
second_peak_hrr_err = [4.3, 22.6, 11.0, 23.8, 20.4, 43.7, 19.3, 29.1, 18.7]

# Second Peak Time (s)
second_peak_time = [194.3, 423.3, 470.8, 465.7, 514.0, 538.0, 626.7, 636.3, 596.6]
second_peak_time_err = [101.5, 63.0, 32.3, 13.3, 28.3, 26.9, 25.3, 9.8, 84.7]

# Calculate time difference from ignition (TTI) to peak
first_peak_time_diff = [first_peak_time[i] - tti[i] for i in range(len(woods))]
second_peak_time_diff = [second_peak_time[i] - tti[i] for i in range(len(woods))]

# Error in time difference (errors are independent, so we add in quadrature)
first_peak_time_diff_err = [np.sqrt(first_peak_time_err[i]**2 + tti_err[i]**2) for i in range(len(woods))]
second_peak_time_diff_err = [np.sqrt(second_peak_time_err[i]**2 + tti_err[i]**2) for i in range(len(woods))]

# Create colormap for density
cmap = plt.cm.viridis
norm = Normalize(vmin=min(density), vmax=max(density))
sm = ScalarMappable(cmap=cmap, norm=norm)

# Create the plot
fig, ax = plt.subplots(figsize=(14, 8))

# Plot each wood type
for i, wood in enumerate(woods):
    color = cmap(norm(density[i]))
    
    # First peak (circles)
    ax.errorbar(first_peak_time_diff[i], first_peak_hrr[i],
                xerr=first_peak_time_diff_err[i], yerr=first_peak_hrr_err[i],
                marker='o', markersize=10, color=color, linestyle='none',
                capsize=5, capthick=2, alpha=0.8, elinewidth=2)
    
    # Second peak (triangles)
    ax.errorbar(second_peak_time_diff[i], second_peak_hrr[i],
                xerr=second_peak_time_diff_err[i], yerr=second_peak_hrr_err[i],
                marker='^', markersize=10, color=color, linestyle='none',
                capsize=5, capthick=2, alpha=0.8, elinewidth=2)

# Labels and title
ax.set_xlabel('Time from Ignition to Peak (s)', fontsize=16)
ax.set_ylabel('Peak HRR (kW/m²)', fontsize=16)

# Add colorbar for density
cbar = plt.colorbar(sm, ax=ax, pad=0.02)
cbar.set_label('Density (g/cm³)', fontsize=16)

# Create custom legend for symbols
from matplotlib.lines import Line2D
legend_elements = [Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', 
                          markersize=10, label='1st Peak', markeredgecolor='black', markeredgewidth=1.5),
                   Line2D([0], [0], marker='^', color='w', markerfacecolor='gray', 
                          markersize=10, label='2nd Peak', markeredgecolor='black', markeredgewidth=1.5)]

ax.legend(handles=legend_elements, loc='upper left', fontsize=14, framealpha=0.95)

plt.tight_layout()
fig.savefig('peaktiming.png', dpi=300, bbox_inches='tight')
plt.show()

# Print the data for reference
print("Wood Type\t\t1st Peak Time Diff (s)\t2nd Peak Time Diff (s)")
print("-" * 70)
for i, wood in enumerate(woods):
    print(f"{wood:15}\t{first_peak_time_diff[i]:6.1f} ± {first_peak_time_diff_err[i]:5.1f}\t\t{second_peak_time_diff[i]:6.1f} ± {second_peak_time_diff_err[i]:5.1f}")