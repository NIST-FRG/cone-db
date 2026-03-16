import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

# Data
heat_flux = np.array([25, 30, 40, 50])
t_ign = np.array([180, 72, 28, 16])

# Calculate 1/sqrt(t_ign)
inv_sqrt_tign = 1 / np.sqrt(t_ign)


# Create figure with two y-axes
fig, ax1 = plt.subplots(figsize=(14, 8))

# Plot t_ign on left y-axis
color1 = '#1f77b4'
ax1.set_xlabel('Heat Flux (kW/m²)', fontsize=16)
ax1.set_ylabel('Time to Ignition, $t_{ign}$ (s)', fontsize=16, color=color1)
ax1.plot(heat_flux, t_ign, 'o-', color=color1, linewidth=2.5, markersize=8, label='$t_{ign}$')
ax1.tick_params(axis='y', labelcolor=color1, labelsize=12)
ax1.tick_params(axis='x', labelsize=12)

# Create second y-axis
ax2 = ax1.twinx()

# Plot 1/sqrt(t_ign) on right y-axis
color2 = '#d62728'
ax2.set_ylabel('$1/\\sqrt{t_{ign}}$ (s$^{-0.5}$)', fontsize=16, color=color2)
ax2.plot(heat_flux, inv_sqrt_tign, 's', color=color2, linewidth=2.5, markersize=8, label='$1/\\sqrt{t_{ign}}$')
ax2.tick_params(axis='y', labelcolor=color2, labelsize=12)

# Add linear fit for 1/sqrt(t_ign) vs heat flux with uncertainty
result = stats.linregress(heat_flux, inv_sqrt_tign)
slope = result.slope
intercept = result.intercept
slope_err = result.stderr
intercept_err = result.intercept_stderr

# Calculate critical heat flux (x-intercept where 1/sqrt(t_ign) = 0)
chf = -intercept / slope

# Propagate uncertainty: q_crit = -b/m
# Using error propagation: σ_q = q_crit * sqrt((σ_b/b)^2 + (σ_m/m)^2)
chf_err = abs(chf) * np.sqrt((intercept_err / intercept)**2 + (slope_err / slope)**2) *2

print(f"Linear fit: 1/sqrt(t_ign) = {slope:.6f} * q'' + {intercept:.6f}")
print(f"Slope: {slope:.6f} ± {slope_err:.6f}")
print(f"Intercept: {intercept:.6f} ± {intercept_err:.6f}")
print(f"R-squared: {result.rvalue**2:.4f}")
print(f"Critical Heat Flux (CHF): {chf:.2f} ± {chf_err:.2f} kW/m²")

# Extend fit line from CHF to beyond max heat flux
heat_flux_fit = np.linspace(chf, heat_flux.max() + 5, 100)
fit_line = slope * heat_flux_fit + intercept
ax2.plot(heat_flux_fit, fit_line, '--', color=color2, linewidth=2.5)

# Mark the critical heat flux on x-axis
ax2.plot(chf, 0, 'v', color=color2, markersize=10)
ax2.annotate(f'$\\dot{{q}}\'\'_{{crit}}$ = {chf:.1f} ± {chf_err:.1f} kW/m²', 
             xy=(chf, 0), 
             xytext=(chf +1.5, 0.03),
             fontsize=14, 
             color=color2,
             ha='right')

# Set x-axis to start at 0
ax1.set_xlim(0, heat_flux.max() + 5)

# Set y-axes to start at 0
ax1.set_ylim(0, None)
ax2.set_ylim(0, None)

# Add legend
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=14)

fig.tight_layout()

fig.savefig('q_crit_RC.png', dpi=300, bbox_inches='tight')
plt.show()