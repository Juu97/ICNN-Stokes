import os
import numpy as np
import pandas as pd
from ICNN import ICNN
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit
import pickle

name = '07'

# Function definitions
def power_law(shearrate, m=300, n=1.483):
    """Power law model for viscosity."""
    return np.log(m * shearrate ** (n - 2))


def carreau(shearrate, mu0=89.8, muinf=3.03, lam=14.2, n=1.483):
    """Carreau (ICNN training, Table_6) model for viscosity."""
    return np.log(muinf + (mu0 - muinf) * (1 + lam * (shearrate) ** 2) ** ((n - 2) / 2))


def calculate_r_squared(y_data, predictions):
    """Calculate R-squared value."""
    mean_y = np.mean(y_data)
    sst = np.sum((y_data - mean_y) ** 2)
    sse = np.sum((y_data - predictions) ** 2)
    return 1 - (sse / sst)


def calculate_mse(y_data, predictions):
    """Calculate Mean Squared Error."""
    return np.mean((y_data - predictions) ** 2)


# Load data
df = pd.read_csv(f"flow_data_{name}.csv")
x = df["shear_rate"].to_numpy()
y = df["viscosity"].to_numpy()

# Load ICNN model
with open(f'icnn_model_{name}.pkl','rb') as f:
    icnn = pickle.load(f)

# -------------- #
# Model comparison
models = [power_law, carreau]
params = {"power_law": [300, 0.483], "carreau": [89.8, 3.03, 14.2, 1.483]}
colors = {'power_law': 'goldenrod', 'carreau': 'red'}

fig, ax = plt.subplots(figsize=(8, 7))
plt.scatter(x, y, label='Data', alpha=0.4, marker='+')

# Fit and plot each model
values = {}
for model in models:
    pf = curve_fit(model, x, np.log(y), params[model.__name__.lower()], maxfev=10000)
    yf = np.exp(model(x, *pf[0]))
    values[model.__name__] = yf
    plt.loglog(x, yf, label=model.__name__, color=colors[model.__name__.lower()])

# Plot ICNN model
plt.loglog(x, icnn.forward(x), label='ICNN', ls='--', color='black', linewidth=1.5)
plt.grid()
plt.legend(fontsize=20)
plt.xlabel('Shear rate', fontsize=14)
plt.ylabel('Viscosity', fontsize=14)
plt.tight_layout()
# plt.savefig(f'NaCL (ICNN training, Table_4){name}.png', format='png', bbox_inches='tight', dpi=300)
plt.show()


# Calculate and print R-squared and MSE for each model
print('\n\n')
print('R2 and MSE:')
for model in models:
    r2 = calculate_r_squared(y, values[model.__name__])
    mse = calculate_mse(y, values[model.__name__])
    print(f'{model.__name__} -> R2 = {r2:.6f}, MSE = {mse:.6f}')

# Calculate and print R-squared and MSE for ICNN model
icnn_r2 = calculate_r_squared(y, icnn.forward(x))
icnn_mse = calculate_mse(y, icnn.forward(x))
print(f'ICNN -> R2 = {icnn_r2:.5f}, MSE = {icnn_mse:.6f}')
print('\n')
