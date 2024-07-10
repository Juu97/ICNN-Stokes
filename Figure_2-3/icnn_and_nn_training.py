import pickle
import pandas as pd
from ICNN import ICNN

# Load data
name = '05'
df = pd.read_csv(f"flow_data_{name}.csv")
x = df["shear_rate"].to_numpy()
y = df["viscosity"].to_numpy()

'''with open('icnn_model.pkl', 'rb') as f:
    icnn = pickle.load(f)'''

# Initialize ICNN model
icnn = ICNN([1, 120 * 6, 56 * 6, 1], activation_function="elu")
standard_nn = ICNN([1, 120 * 6, 56 * 6, 1], activation_function="elu")

number_of_epochs = 300000
lr = 0.01

# Train the ICNN model
icnn.convex_training(x, y, learning_rate=lr, epochs=number_of_epochs, epsilon=30)

# Train the NN standard model
standard_nn.convex_training(x, y, learning_rate=lr, epochs=number_of_epochs, do_convex_training=False)

# Save the ICNN model as pickle
with open(f'icnn_model.pkl', 'wb') as f:
    pickle.dump(icnn, f)

# Save the NN model as pickle
with open(f'standard_nn_model.pkl', 'wb') as f:
    pickle.dump(standard_nn, f)