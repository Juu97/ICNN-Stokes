import os
import pickle

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit

np.seterr(invalid='ignore')

name = '00'
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
df = pd.read_csv(f"data/flow_data_NACL{name}.csv")
x = df["shear_rate"].to_numpy()
y = df["viscosity"].to_numpy()

# -------------- #

with open(f'icnn_model_{name}.pkl', 'rb') as f:
    icnn = pickle.load(f)

# Calculate and print R-squared and MSE for ICNN model
icnn_r2 = calculate_r_squared(y, icnn.forward(x))
icnn_mse = calculate_mse(y, icnn.forward(x))
print(f'ICNN -> R2 = {icnn_r2:.6f}, MSE = {icnn_mse:.6f}')
print('\n')