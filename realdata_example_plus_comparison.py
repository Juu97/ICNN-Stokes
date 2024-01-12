import os
import numpy as np
import pandas as pd
from ICNN import ICNN
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit


# Function definitions
def power_law(shearrate, m=300, n=0.483):
    """Power law model for viscosity."""
    return np.log(m * shearrate ** (n - 1 / 2))


def carreau(shearrate, mu0=89.8, muinf=3.03, lam=14.2, n=0.483):
    """Carreau model for viscosity."""
    return np.log(muinf + (mu0 - muinf) / (1 + (lam * shearrate) ** 2) ** ((1 - n) / 2))


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
df = pd.read_csv("flow_data.csv")
x = df["shear_rate"].to_numpy()
y = df["viscosity"].to_numpy()

# Initialize ICNN model
icnn = ICNN([1, 120, 56, 1], activation_function="elu")
number_of_epochs = 75000
lr = 0.01

# Train the ICNN model
icnn.convex_training(x, y, learning_rate=lr, epochs=number_of_epochs, epsilon=30)

# Plot training data and ICNN model predictions
plt.scatter(x, y, s=10, label='Training Data', alpha=0.5)
plt.loglog(x, icnn(x), label='ICNN', color="black", ls='--')
plt.legend()
plt.grid()
plt.show()

# -------------- #
# Model comparison
models = [power_law, carreau]
params = {"power_law": [300, 0.483], "carreau": [89.8, 3.03, 14.2, 0.483]}
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
print(f'ICNN -> R2 = {icnn_r2:.6f}, MSE = {icnn_mse:.6f}')
print('\n')
