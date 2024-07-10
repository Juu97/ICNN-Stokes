import pickle
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle, ConnectionPatch

with open('icnn_model.pkl', 'rb') as f:
    icnn = pickle.load(f)

with open('standard_nn_model.pkl', 'rb') as f:
    standard_nn = pickle.load(f)

name = '05'
df = pd.read_csv(f"flow_data_{name}.csv")
x = df["shear_rate"].to_numpy()
y = df["viscosity"].to_numpy()

shear_rate = np.logspace(np.log10(x.min()), np.log10(x.max()), 1000)
shear_rate2 = np.linspace(x.min(), x.max(), 1000)

# Combine both arrays, ensuring they cover the entire range from min_x to max_x without overlap
shear_rate = np.sort(np.concatenate([shear_rate, shear_rate2]))
# Compute values
nn_values = standard_nn(shear_rate)
icnn_values = icnn(shear_rate)

# Create the subplot 1x3
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# First plot (linear scale)
ax1.plot(shear_rate, icnn_values, label='ICNN(t)', color='red')
ax1.plot(shear_rate, nn_values, '--', label='NN(t)', color='blue')
ax1.scatter(x, y, label='Data', color='green', s=5, alpha=0.5)
ax1.set_xlabel('Shear rate', fontsize=14)
ax1.set_ylabel(r'$k(t)$', fontsize=14)
ax1.legend(fontsize=14)
ax1.grid(True)

# Second plot (log-log scale)
ax2.plot(shear_rate, icnn_values, label='ICNN(t)', color='red')
ax2.plot(shear_rate, nn_values, '--', label='NN(t)', color='blue')
ax2.scatter(x, y, label='Data', color='green', s=5, alpha=0.5)
ax2.set_xlabel('Shear rate', fontsize=14)
ax2.set_ylabel(r'$k(t)$', fontsize=14)
ax2.legend(fontsize=14)
ax2.set_xscale('log')
ax2.set_yscale('log')
ax2.grid(True, which="both", ls="--")


# Zoomed inset
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

axins = inset_axes(ax2, width="30%", height="30%", loc='lower left', borderpad=3)
axins.loglog(shear_rate, icnn_values, color='red')
axins.loglog(shear_rate, nn_values, '--', color='blue')
axins.scatter(x, y, color='green', s=5,alpha=0.5)
axins.set_xlim(249, 1100)
axins.set_ylim(0.004, 0.007)
axins.grid(True, which="both", ls="--")

# Remove tick labels
axins.tick_params(axis='both', which='both', labelleft=False, labelbottom=False)

# Add a gray square where the zoom is taken
rect = Rectangle((249, 0.004), 802, 0.003, linewidth=1, edgecolor='gray', facecolor='none')
ax2.add_patch(rect)

# Connecting lines to the limits of axins
xy1 = (249, 0.004)
xy2 = (1052, 0.004)
xy3 = (249, 0.007)
xy4 = (1052, 0.007)

con1 = ConnectionPatch(xyA=xy1, coordsA=axins.transData, xyB=xy1, coordsB=ax2.transData, color='gray')
con2 = ConnectionPatch(xyA=xy2, coordsA=axins.transData, xyB=xy2, coordsB=ax2.transData, color='gray')
con3 = ConnectionPatch(xyA=xy3, coordsA=axins.transData, xyB=xy3, coordsB=ax2.transData, color='gray')
con4 = ConnectionPatch(xyA=xy4, coordsA=axins.transData, xyB=xy4, coordsB=ax2.transData, color='gray')

ax2.add_artist(con1)
ax2.add_artist(con2)
ax2.add_artist(con3)
ax2.add_artist(con4)


# Increase font size of ticks for all subplots
for ax in (ax1, ax2):
    ax.tick_params(axis='both', which='major', labelsize=12)

plt.tight_layout()
plt.show()


# Third plot (linear scale, multiplied by shear rate)
plt.plot(shear_rate, icnn_values * shear_rate, label='ICNN(t)*t', color='red', linewidth=2)
plt.plot(shear_rate, nn_values * shear_rate, '--', label='NN(t)*t', color='blue', linewidth=2)
plt.xlabel('Shear rate', fontsize=14)
plt.ylabel(r'$k(t) * t$', fontsize=14)
plt.legend(fontsize=14)
plt.tight_layout()
plt.grid(True)
plt.show()