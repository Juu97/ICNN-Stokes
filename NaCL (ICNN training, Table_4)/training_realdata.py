import pickle

import pandas as pd
from ICNN import ICNN
import matplotlib.pyplot as plt

# Load data
name = '00'
print('Training for', name)
df = pd.read_csv(f"flow_data_{name}.csv")
x = df["shear_rate"].to_numpy()
y = df["viscosity"].to_numpy()

# Initialize ICNN model
icnn = ICNN([1, 120, 56, 1], activation_function="elu")
number_of_epochs = 100000
lr = 0.01

# Train the ICNN model
icnn.convex_training(x, y, learning_rate=lr, epochs=number_of_epochs, epsilon=30)

# Plot ICNN model
plt.loglog(x, icnn(x), label='ICNN 00', ls='--', color='black', linewidth=1.5)
plt.scatter(x, y, label='Data', color='blue', s=2)
plt.grid()
plt.legend()
plt.xlabel('Shear rate')
plt.ylabel('Viscosity')
plt.show()

# Save the model as pickle
with open(f'icnn_model_{name}.pkl', 'wb') as f:
    pickle.dump(icnn, f)

