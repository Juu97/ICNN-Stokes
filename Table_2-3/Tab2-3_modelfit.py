import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit

np.seterr(invalid='ignore')

# Function definitions
def power_law(shearrate, m=300, n=1.483):
    """Power law model for viscosity."""
    return np.log(m * shearrate ** (n - 2))

def carreau(shearrate, mu0=89.8, muinf=3.03, lam=14.2, n=1.483):
    """Carreau model for viscosity."""
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
df = pd.read_csv("data/flow_data_NACL00.csv")
x = df["shear_rate"].to_numpy()
y = df["viscosity"].to_numpy()

# -------------- #
# Model comparison
models = [power_law, carreau]
params = {"power_law": [300, 1.483], "carreau": [0.6, 0.01, 100, 1.42]}
colors = {'power_law': 'goldenrod', 'carreau': 'red'}

fig, ax = plt.subplots(figsize=(8, 7))

# Fit and plot each model
values = {}
for model in models:
    pf = curve_fit(model, x, np.log(y), params[model.__name__.lower()], maxfev=10000)
    yf = np.exp(model(x, *pf[0]))
    values[model.__name__] = yf
    plt.scatter(x, y, label='Data', alpha=0.4, marker='+')
    plt.loglog(x, yf, label=model.__name__, color=colors[model.__name__.lower()])
    print(f'{model.__name__} -> param: {pf[0]}')
    r2 = calculate_r_squared(y, values[model.__name__])
    mse = calculate_mse(y, values[model.__name__])
    print(f'{model.__name__} -> R2 = {r2:.6f}, MSE = {mse:.6f}')
    
